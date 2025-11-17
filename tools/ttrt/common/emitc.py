# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import signal
import io
import subprocess
import time
import shutil
import traceback

from ttrt.common.util import *
from ttrt.common.run import Run


class EmitC:
    registered_args = {}

    @staticmethod
    def initialize_api():
        EmitC.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        EmitC.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        EmitC.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        EmitC.register_arg(
            name="--program-index",
            type=str,
            default="all",
            choices=["all"] + [str(i) for i in range(0, 5)],
            help="the program inside the fbb to run",
        )
        EmitC.register_arg(
            name="--loops",
            type=int,
            default=1,
            choices=None,
            help="number of loops",
        )
        EmitC.register_arg(
            name="--result-file",
            type=str,
            default="emitc_results.json",
            choices=None,
            help="test file to save results to",
        )
        EmitC.register_arg(
            name="--disable-golden",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable golden comparison for intermediate and output tensors",
        )
        EmitC.register_arg(
            name="--flatbuffer",
            type=str,
            default="",
            choices=None,
            help="Provide a file or directory path for flatbuffer binary files to compare outputs to",
        )
        EmitC.register_arg(
            name="--memory",
            type=bool,
            default=False,
            choices=[True, False],
            help="dump memory reports after every op execution",
        )
        EmitC.register_arg(
            name="--disable-eth-dispatch",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable putting dispatch on ethernet cores - place it on worker cores instead",
        )
        EmitC.register_arg(
            name="--ignore-version",
            type=bool,
            default=False,
            choices=[True, False],
            help="Ignore check for Major/Minor/Patch between flatbuffer and TTRT, use at your own risk.",
        )
        EmitC.register_arg(
            name="--enable-program-cache",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable program cache in ttnn runtime",
        )
        EmitC.register_arg(
            name="--trace-region-size",
            type=int,
            default=0,
            choices=None,
            help="Device trace region size",
        )
        EmitC.register_arg(
            name="--dump-device-rate",
            type=int,
            default=1000,
            choices=None,
            help="Rate at which to flush device perf information",
        )
        EmitC.register_arg(
            name="--benchmark",
            type=bool,
            default=False,
            choices=[True, False],
            help="Enable benchmark mode with warmup and e2e time measurements (automatically enables program cache)",
        )
        EmitC.register_arg(
            name="dylib",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        EmitC.register_arg(
            name="--disable-ttrt-callbacks",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable ttrt callbacks",
        )
        EmitC.register_arg(
            name="--print-input-output-tensors",
            type=bool,
            default=False,
            choices=[True, False],
            help="print input and output tensors",
        )
        EmitC.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in EmitC.registered_args.items():
            if type(args) == dict:
                if name in args.keys():
                    self[name] = args[name]
                else:
                    self[name] = attributes["default"]
            else:
                converted_name = name
                if name != "dylib":
                    converted_name = converted_name.lstrip("-")
                    converted_name = converted_name.replace("-", "_")
                self[name] = getattr(args, converted_name)

        self.logger = logger if logger != None else Logger(self["--log-file"])
        self.logging = self.logger.get_logger()
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
        self.emitc_dylibs = []
        self.ttnn_binaries = {}
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing emitc API")

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        if self["--save-artifacts"]:
            self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing emitc API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for emitc API")

        emitc_dylib_paths = self.file_manager.find_emitc_dylib_paths(self["dylib"])

        self.logging.debug(f"emitc_dylib_paths={emitc_dylib_paths}")

        for path in emitc_dylib_paths:
            dylib = EmitCDylib(self.logger, self.file_manager, path)
            self.emitc_dylibs.append(dylib)
            if self["--flatbuffer"]:
                if os.path.isdir(self["--flatbuffer"]):
                    corresponding_ttnn_path = (
                        self.file_manager.find_so_corresponding_ttnn_in_directory(
                            path, self["--flatbuffer"]
                        )
                    )
                    if corresponding_ttnn_path is None:
                        self.logging.warning(
                            f"SKIP: no ttnn file found corresponding to dylib ={path} in directory={self['--flatbuffer']}"
                        )
                        continue
                else:
                    corresponding_ttnn_path = self["--flatbuffer"]
                self.logging.debug(
                    f"Found ttnn file corresponding to dylib ={corresponding_ttnn_path}"
                )
                bin = Binary(self.logger, self.file_manager, corresponding_ttnn_path)
                try:
                    bin.check_version(ignore=self["--ignore-version"])
                    self.ttnn_binaries[dylib] = bin
                except Exception as e:
                    self.logging.warning(
                        f"SKIP: dylib comparison for test={path} was skipped with exception={str(e)}"
                    )
                    continue

        self.logging.debug(f"------finished checking constraints for emitc API")

    def execute(self):
        import ttrt.runtime

        self.logging.debug(f"------executing emitc API")

        if len(self.emitc_dylibs) == 0:
            self.logging.warning(f"no EmitC dylibs found to run - returning early")
            return

        # Initialize `device` to `None` for error handling in case device opening fails
        device = None

        for dylib in self.emitc_dylibs:
            # Open the dylib
            emitc_dylib_handle = ttrt.runtime.test.open_so(dylib.file_path)
            self.logging.debug(f"opened emitc dylib={dylib.file_path}")
            try:
                compare_to_ttnn = False
                if dylib in self.ttnn_binaries:
                    bin = self.ttnn_binaries[dylib]
                    compare_to_ttnn = True

                    command_options = f"--program-index {self['--program-index']} --loops {self['--loops']} --save-artifacts "

                    if self["--artifact-dir"]:
                        command_options += f" --artifact-dir {self['--artifact-dir']} "

                    if self["--memory"]:
                        command_options += " --memory "

                    if self["--disable-eth-dispatch"]:
                        command_options += " --disable-eth-dispatch "

                    if self["--disable-golden"]:
                        command_options += " --disable-golden "

                    if self["--enable-program-cache"]:
                        command_options += " --enable-program-cache "

                    if self["--trace-region-size"]:
                        command_options += " --trace-region-size "

                    if self["--dump-device-rate"] != 1000:
                        command_options += (
                            f" --dump-device-rate {self['--dump-device-rate']} "
                        )

                    if self["--benchmark"]:
                        command_options += " --benchmark "

                    if self["--ignore-version"]:
                        command_options += " --ignore-version "

                    if self["--disable-ttrt-callbacks"]:
                        command_options += " --disable-ttrt-callbacks "

                    if self["--print-input-output-tensors"]:
                        command_options += " --print-input-output-tensors "

                    ttrt_executable_path = shutil.which("ttrt")
                    test_command = (
                        f"{ttrt_executable_path} run {bin.file_path} {command_options}"
                    )
                    self.logging.info(
                        f"test command for binary={bin.file_path} is: {test_command}"
                    )
                    testProcess = subprocess.Popen(
                        [test_command],
                        shell=True,
                        preexec_fn=os.setsid,
                    )

                    def signal_handler(sig, frame):
                        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                        sys.exit(3)

                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
                    testProcess.communicate()

                # Open a device of default shape
                dispatch_core_type = ttrt.runtime.DispatchCoreType.ETH

                if self["--disable-eth-dispatch"]:
                    dispatch_core_type = ttrt.runtime.DispatchCoreType.WORKER
                mesh_options = ttrt.runtime.MeshDeviceOptions()
                mesh_options.dispatch_core_type = dispatch_core_type
                mesh_options.enable_program_cache = self["--enable-program-cache"]
                mesh_options.trace_region_size = self["--trace-region-size"]

                if compare_to_ttnn:
                    # Open a device of shape (x,y), where (x,y) is the mesh shape supplied by the flatbuffer
                    fb_mesh_shape = bin.get_program(0).mesh_shape
                    mesh_options.mesh_shape = fb_mesh_shape

                device = ttrt.runtime.open_mesh_device(mesh_options)

                # Run to EmitC
                program_names = ttrt.runtime.test.get_so_programs(
                    emitc_dylib_handle, dylib.file_path
                )

                self.logging.debug(f"Program names found: {program_names}")

                if self["--program-index"] != "all":
                    if len(program_names) > int(self["--program-index"]):
                        self.logging.warning(
                            f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        )
                        continue

                if compare_to_ttnn:
                    # Load input and output tensors for each program from artifacts
                    fbb_run_directory = self.artifacts.get_binary_run_folder_path(bin)
                    fbb_torch_inputs = self.file_manager.load_tensors_from_artifacts(
                        bin, "input", fbb_run_directory
                    )
                    fbb_torch_outputs = self.file_manager.load_tensors_from_artifacts(
                        bin,
                        "device_output",
                        fbb_run_directory,
                    )

                for program_index, program_name in enumerate(program_names):
                    if self["--program-index"] != "all" and program_index != int(
                        self["--program-index"]
                    ):
                        continue
                    self.logging.debug(
                        f"evaluating program={program_name} for file={dylib.file_path}"
                    )
                    emitc_artifact_path = f"{self.artifacts.get_emitc_dylib_folder_path(dylib)}/program_{program_index}"

                    if compare_to_ttnn:
                        fbb_runtime_inputs = []

                        for i, fbb_torch_input in enumerate(
                            fbb_torch_inputs["program_" + str(program_index)]
                        ):
                            new_input = create_tensor(fbb_torch_input)
                            fbb_runtime_inputs.append(new_input)

                            if self["--save-artifacts"]:
                                self.artifacts.save_torch_tensor(
                                    emitc_artifact_path,
                                    fbb_torch_input,
                                    f"emitc_input_{i}.pt",
                                )

                            if self["--print-input-output-tensors"]:
                                self.logging.info(
                                    f"Input tensor {i}: {fbb_torch_input}"
                                )

                        # pre-upload inputs
                        emitc_runtime_inputs = convert_input_layouts(
                            device,
                            fbb_runtime_inputs,
                            bin.fbb,
                            program_index,
                        )
                    else:
                        emitc_runtime_inputs = ttrt.runtime.test.create_inputs(
                            emitc_dylib_handle,
                            program_name,
                            device,
                            dylib.file_path,
                        )
                        emitc_torch_inputs = []
                        if (
                            self["--save-artifacts"]
                            or self["--print-input-output-tensors"]
                            or compare_to_ttnn
                        ):
                            for i, emitc_runtime_input in enumerate(
                                emitc_runtime_inputs
                            ):
                                # Ensure inputs are converted to host and untilized before tensor conversion
                                host_input = ttrt.runtime.to_host(
                                    emitc_runtime_input, untilize=True
                                )[0]
                                emitc_torch_input = convert_runtime_to_torch_tensor(
                                    host_input
                                )
                                emitc_torch_inputs.append(emitc_torch_input)

                                if self["--save-artifacts"]:
                                    self.artifacts.save_torch_tensor(
                                        emitc_artifact_path,
                                        emitc_torch_input,
                                        f"emitc_input_{i}.pt",
                                    )

                                if self["--print-input-output-tensors"]:
                                    self.logging.info(
                                        f"Input tensor {i}: {emitc_torch_input}"
                                    )

                    for loop in range(self["--loops"]):
                        emitc_runtime_outputs = ttrt.runtime.test.run_so_program(
                            emitc_dylib_handle,
                            program_name,
                            emitc_runtime_inputs,
                            device,
                        )
                        emitc_runtime_outputs = [
                            ttrt.runtime.to_host(emitc_out, untilize=True)[0]
                            for emitc_out in emitc_runtime_outputs
                        ]

                    emitc_torch_outputs = []
                    if (
                        self["--save-artifacts"]
                        or self["--print-input-output-tensors"]
                        or compare_to_ttnn
                    ):
                        for i, emitc_runtime_output in enumerate(emitc_runtime_outputs):
                            # Ensure outputs are converted to host and untilized before tensor conversion
                            host_output = ttrt.runtime.to_host(
                                emitc_runtime_output, untilize=True
                            )[0]
                            emitc_torch_output = convert_runtime_to_torch_tensor(
                                host_output
                            )
                            emitc_torch_outputs.append(emitc_torch_output)

                            if self["--save-artifacts"]:
                                self.artifacts.save_torch_tensor(
                                    emitc_artifact_path,
                                    emitc_torch_output,
                                    f"emitc_output_{i}.pt",
                                )

                            if self["--print-input-output-tensors"]:
                                self.logging.info(
                                    f"Output tensor {i}: {emitc_torch_output}"
                                )

                        if compare_to_ttnn:
                            fbb_runtime_outputs = []
                            for fbb_torch_output in fbb_torch_outputs[
                                "program_" + str(program_index)
                            ]:
                                new_output = create_tensor(fbb_torch_output)
                                fbb_runtime_outputs.append(new_output)

                            self.logging.debug(
                                f"got emitc outputs for program_index={program_index}, loop={loop}"
                            )

                            all_tensors_match = ttrt.runtime.test.compare_outs(
                                fbb_runtime_outputs, emitc_runtime_outputs
                            )

                            if not all_tensors_match:
                                self.logging.error(
                                    "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                )
                                self.logging.error(
                                    fbb_torch_outputs, emitc_torch_outputs
                                )
                                raise Exception(
                                    "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                )
                            self.logging.info(
                                f"EmitC tensors match for {bin.file_path}"
                            )
            except Exception as e:
                result = "error"
                if isinstance(e, TTRTTestException):
                    result = "test_error"
                test_result = {
                    "file_path": dylib.file_path,
                    "result": result,
                    "exception": str(e),
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                }
                self.logging.error(
                    f"ERROR: test={dylib.file_path} experienced an error with exception={str(e)}"
                )
                self.results.add_result(test_result)
                dylib.test_result = result
                traceback.print_exc()
                continue
            finally:
                # Only close the device it if was opened
                if device is not None:
                    ttrt.runtime.close_mesh_device(device)
                    device = None

                # Clean up memory, avoid Python GC calling __del__ on objects from closed so
                import gc

                del emitc_runtime_inputs, emitc_runtime_outputs
                try:
                    del emitc_torch_inputs
                except UnboundLocalError:
                    pass
                try:
                    del emitc_torch_outputs
                except UnboundLocalError:
                    pass
                gc.collect()

                ttrt.runtime.test.close_so(emitc_dylib_handle)

        self.logging.debug(f"finished executing emitc_dylibs")

        self.logging.debug(f"------finished executing emitc API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing emitc API")

        for dylib in self.emitc_dylibs:
            if self["--save-artifacts"]:
                self.artifacts.save_emitc_dylib(dylib)

            if dylib.test_result == "pass":
                test_result = {
                    "file_path": dylib.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                }
                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={dylib.file_path}")
            else:
                self.logging.error(f"ERROR: test case={dylib.file_path}")

        self.results.save_results(self["--result-file"])

        self.logging.debug(f"------finished postprocessing emitc API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting emitc API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished emitc API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

    @staticmethod
    def register_arg(name, type, default, choices, help):
        EmitC.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        emitc_parser = subparsers.add_parser(
            "emitc",
            help="run EmitC Dylib tests and optionally compare outputs to flatbuffer outputs",
        )
        emitc_parser.set_defaults(api=EmitC)

        for name, attributes in EmitC.registered_args.items():
            if name == "dylib":
                emitc_parser.add_argument(f"{name}", help=attributes["help"])
            elif attributes["type"] == bool:
                emitc_parser.add_argument(
                    f"{name}",
                    action="store_true",
                    help=attributes["help"],
                )
            else:
                emitc_parser.add_argument(
                    f"{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=attributes["choices"],
                    help=attributes["help"],
                )

        return emitc_parser
