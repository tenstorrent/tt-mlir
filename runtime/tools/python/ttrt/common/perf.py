# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import shutil
import atexit

from ttrt.common.util import *
from ttrt.common.query import Query


class Perf:
    registered_args = {}

    @staticmethod
    def initialize_api():
        Perf.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        Perf.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        Perf.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        Perf.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        Perf.register_arg(
            name="--program-index",
            type=str,
            default="all",
            choices=["all"] + [str(i) for i in range(0, 5)],
            help="the program inside the fbb to run",
        )
        Perf.register_arg(
            name="--loops",
            type=int,
            default=1,
            choices=None,
            help="number of loops",
        )
        Perf.register_arg(
            name="--host-only",
            type=bool,
            default=False,
            choices=[True, False],
            help="collect performance trace on host only",
        )
        Perf.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in Perf.registered_args.items():
            if type(args) == dict:
                if name in args.keys():
                    self[name] = args[name]
                else:
                    self[name] = attributes["default"]
            else:
                # argument got parsed to hyphen's for underscrolls and leading hyphen's removed - need to put back
                converted_name = name
                if name != "binary":
                    converted_name = converted_name.lstrip("-")
                    converted_name = converted_name.replace("-", "_")
                self[name] = getattr(args, converted_name)

        self.logger = logger if logger != None else Logger(self["--log-file"])
        self.logging = self.logger.get_logger()
        self.globals = Globals(self.logger)
        self.file_manager = FileManager(self.logger)
        self.artifacts = (
            artifacts
            if artifacts != None
            else Artifacts(
                self.logger,
                self.file_manager,
                artifacts_folder_path=self["--artifact-dir"],
            )
        )
        self.query = Query({"--quiet": True}, self.logger, self.artifacts)
        self.ttnn_binaries = []
        self.ttmetal_binaries = []
        self.tracy_capture_tool_path = (
            f"{self.globals.get_ttmetal_home_path()}/capture-release"
        )
        self.tracy_csvexport_tool_path = (
            f"{self.globals.get_ttmetal_home_path()}/csvexport-release"
        )
        self.tracy_capture_tool_process = None
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing perf API")
        self.query()

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing perf API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for perf API")

        assert self.file_manager.check_file_exists(
            self.tracy_capture_tool_path
        ), f"perf tool={self.tracy_capture_tool_path} does not exist - rebuild using perf mode"
        assert self.file_manager.check_file_exists(
            self.tracy_csvexport_tool_path
        ), f"perf tool={self.tracy_csvexport_tool_path} does not exist - rebuild using perf mode"

        if not hasattr(self, "binary"):
            # load from Capsule instead. only TTNN Path is supported for now
            bin = Binary(self.logger, self.file_manager, "", self["--capsule"])
            if not bin.check_version():
                self.logger.warning(
                    "Flatbuffer version not present, are you sure that the binary is valid? - Skipped"
                )
                return

            if not bin.check_system_desc(self.query):
                self.logger.warning(
                    "System desc does not match, are you sure that the binary is valid? - Skipped"
                )
                return

            if self["--program-index"] != "all":
                if not bin.check_program_index_exists(int(self["--program-index"])):
                    self.logging.warning(
                        f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                    )
                    return
            self.ttnn_binaries.append(bin)
        else:
            ttnn_binary_paths = self.file_manager.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = self.file_manager.find_ttmetal_binary_paths(
                self["binary"]
            )

            self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")
            self.logging.debug(f"ttmetal_binary_paths={ttmetal_binary_paths}")

            for path in ttnn_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                try:
                    bin.check_version()
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.results.add_result(test_result)
                    continue

                try:
                    bin.check_system_desc(self.query)
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.results.add_result(test_result)
                    continue

                if self["--program-index"] != "all":
                    if not bin.check_program_index_exists(int(self["--program-index"])):
                        message = f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        self.logging.warning(message)
                        test_result = {
                            "file_path": path,
                            "result": "skip",
                            "exception": message,
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "program_index": self["--program-index"],
                        }
                        self.results.add_result(test_result)
                        continue

                self.ttnn_binaries.append(bin)

            self.logging.debug(f"finished checking constraints for run API")

            for path in ttmetal_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                try:
                    bin.check_version()
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.results.add_result(test_result)
                    continue

                try:
                    bin.check_system_desc(self.query)
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.results.add_result(test_result)
                    continue

                if self["--program-index"] != "all":
                    if not bin.check_program_index_exists(int(self["--program-index"])):
                        message = f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        self.logging.warning(message)
                        test_result = {
                            "file_path": path,
                            "result": "skip",
                            "exception": message,
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "program_index": self["--program-index"],
                        }
                        self.results.add_result(test_result)
                        continue

                self.ttmetal_binaries.append(bin)

        self.logging.debug(f"------finished checking constraints for perf API")

    def execute(self):
        self.logging.debug(f"------executing perf API")

        sys.path.append(f"{get_ttrt_metal_home_path()}")
        sys.path.append(f"{get_ttrt_metal_home_path()}/ttnn")

        import tracy
        import tracy.tracy_state
        from tt_metal.tools.profiler.process_ops_logs import process_ops
        from tt_metal.tools.profiler.common import (
            PROFILER_LOGS_DIR,
            TRACY_OPS_TIMES_FILE_NAME,
            TRACY_OPS_DATA_FILE_NAME,
            TRACY_FILE_NAME,
            PROFILER_DEVICE_SIDE_LOG,
        )

        self.file_manager.remove_directory(PROFILER_LOGS_DIR)
        self.file_manager.create_directory(PROFILER_LOGS_DIR)

        def get_available_port():
            ip = socket.gethostbyname(socket.gethostname())

            for port in range(8086, 8500):
                try:
                    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    serv.bind((ip, port))
                    return str(port)
                except PermissionError as e:
                    pass
                except OSError as e:
                    pass
            return None

        def generate_report(binary_perf_folder):
            child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
            time_out = 15
            time_count = 0
            while not os.path.exists(PROFILER_LOGS_DIR / TRACY_FILE_NAME):
                if time_count > time_out:
                    raise Exception(
                        f"tracy capture output file {PROFILER_LOGS_DIR / TRACY_FILE_NAME} was not generated"
                    )
                time_count += 1
                time.sleep(1)

            with open(PROFILER_LOGS_DIR / TRACY_OPS_TIMES_FILE_NAME, "w") as csv_file:
                child_call_str = f"-x {','.join(child_calls)}"
                subprocess.run(
                    f"{self.tracy_csvexport_tool_path} -u -p TT_DNN {child_call_str} {PROFILER_LOGS_DIR / TRACY_FILE_NAME}",
                    shell=True,
                    check=True,
                    stdout=csv_file,
                    stderr=subprocess.DEVNULL,
                )

            with open(PROFILER_LOGS_DIR / TRACY_OPS_DATA_FILE_NAME, "w") as csv_file:
                subprocess.run(
                    f'{self.tracy_csvexport_tool_path} -m -s ";" {PROFILER_LOGS_DIR / TRACY_FILE_NAME}',
                    shell=True,
                    check=True,
                    stdout=csv_file,
                    stderr=subprocess.DEVNULL,
                )
            process_ops(binary_perf_folder, "", True)

        def save_perf_artifacts(binary_perf_folder):
            tracy_ops_times_file = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_OPS_TIMES_FILE_NAME}"
            tracy_ops_data_file = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_OPS_DATA_FILE_NAME}"
            tracy_file = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_FILE_NAME}"

            try:
                self.file_manager.check_file_exists(tracy_ops_times_file)
                self.file_manager.check_file_exists(tracy_ops_data_file)
                self.file_manager.check_file_exists(tracy_file)
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

            for root, dirs, files in os.walk(binary_perf_folder, topdown=False):
                if root == binary_perf_folder:
                    continue

                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    dest_path = os.path.join(binary_perf_folder, file_name)
                    self.file_manager.copy_file(dest_path, file_path)

                if os.listdir(root):
                    self.file_manager.remove_directory(root)

        def _execute(binaries):
            if len(binaries) == 0:
                self.logging.warning(f"no binaries found to run - returning early")
                return

            for bin in binaries:
                try:
                    port = get_available_port()

                    if not port:
                        raise Exception("No available port found")

                    env_vars = dict(os.environ)
                    env_vars["TRACY_PORT"] = port

                    if not self["--host-only"]:
                        env_vars["TT_METAL_CLEAR_L1"] = "1"
                        env_vars["TT_METAL_DEVICE_PROFILER"] = "1"
                        env_vars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "1"

                    current_pythonpath = env_vars.get("PYTHONPATH", "")
                    path1 = f"{get_ttrt_metal_home_path()}"
                    path2 = f"{get_ttrt_metal_home_path()}/ttnn"
                    new_pythonpath = (
                        f"{current_pythonpath}{os.pathsep}{path1}{os.pathsep}{path2}"
                    )
                    env_vars["PYTHONPATH"] = new_pythonpath

                    tracy_capture_tool_command = f"{self.tracy_capture_tool_path} -o {PROFILER_LOGS_DIR / TRACY_FILE_NAME} -f -p {port}"
                    self.tracy_capture_tool_process = subprocess.Popen(
                        tracy_capture_tool_command, shell=True
                    )

                    command_options = f"--program-index {self['--program-index']} --loops {self['--loops']}"

                    library_link_path = self.globals.get_ld_path(
                        f"{self.globals.get_ttmetal_home_path()}"
                    )
                    test_env_flags = f"LD_LIBRARY_PATH={library_link_path}"

                    ttrt_executable_path = shutil.which("ttrt")
                    test_command = f"{test_env_flags} python -m tracy -p {ttrt_executable_path} run {bin.file_path} --save-artifacts {command_options}"
                    print(f"test command for binary={bin.file_path} is: {test_command}")
                    testProcess = subprocess.Popen(
                        [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
                    )

                    def signal_handler(sig, frame):
                        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                        self.tracy_capture_tool_process.terminate()
                        self.tracy_capture_tool_process.communicate()
                        sys.exit(3)

                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
                    testProcess.communicate()

                    try:
                        self.tracy_capture_tool_process.communicate(timeout=15)
                        if not self["--host-only"]:
                            self.file_manager.copy_file(
                                PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG,
                                f"{os.getcwd()}/generated/profiler/.logs/{PROFILER_DEVICE_SIDE_LOG}",
                            )
                        generate_report(self.artifacts.get_binary_perf_folder_path(bin))
                    except subprocess.TimeoutExpired as e:
                        self.tracy_capture_tool_process.terminate()
                        self.tracy_capture_tool_process.communicate()
                        raise Exception(
                            f"No profiling data could be captured. Please make sure you are on the correct build"
                        )

                    save_perf_artifacts(self.artifacts.get_binary_perf_folder_path(bin))
                except Exception as e:
                    test_result = {
                        "file_path": bin.file_path,
                        "result": "error",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.results.add_result(test_result)
                    bin.test_result = "error"
                    continue

        self.logging.debug(f"executing ttnn binaries")
        _execute(self.ttnn_binaries)
        self.logging.debug(f"finished executing ttnn binaries")

        self.logging.debug(f"executing ttmetal binaries")
        _execute(self.ttmetal_binaries)
        self.logging.debug(f"finished executing ttmetal binaries")

        self.logging.debug(f"------finished executing perf API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing perf API")

        if self["--save-artifacts"]:
            for bin in self.ttnn_binaries:
                self.artifacts.save_binary(bin, self.query)

            for bin in self.ttmetal_binaries:
                self.artifacts.save_binary(bin, self.query)

        for bin in self.ttnn_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                }
                self.results.add_result(test_result)

        for bin in self.ttmetal_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                }
                self.results.add_result(test_result)

        self.results.save_results("perf_results.json")

        self.logging.debug(f"------finished postprocessing perf API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting perf API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished perf API----------------------------"
        )

    @staticmethod
    def register_arg(name, type, default, choices, help):
        Perf.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        perf_parser = subparsers.add_parser(
            "perf", help="run performance trace and collect performance data"
        )
        perf_parser.set_defaults(api=Perf)

        for name, attributes in Perf.registered_args.items():
            if name == "binary":
                perf_parser.add_argument(f"{name}", help=attributes["help"])
            elif attributes["type"] == bool:
                perf_parser.add_argument(
                    f"{name}",
                    action="store_true",
                    help=attributes["help"],
                )
            else:
                perf_parser.add_argument(
                    f"{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=attributes["choices"],
                    help=attributes["help"],
                )

        return perf_parser
