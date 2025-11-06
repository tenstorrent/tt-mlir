# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import sys
import signal
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import shutil
import atexit
import traceback
from pathlib import Path
import csv
import ast

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
            name="--port",
            type=int,
            default=0,
            choices=None,
            help="port to run tracy client server application",
        )
        Perf.register_arg(
            name="--result-file",
            type=str,
            default="perf_results.json",
            choices=None,
            help="test file to save results to",
        )
        Perf.register_arg(
            name="--disable-golden",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable golden comparison for intermediate and output tensors",
        )
        Perf.register_arg(
            name="--memory",
            type=bool,
            default=False,
            choices=[True, False],
            help="dump memory reports after every op execution",
        )
        Perf.register_arg(
            name="--disable-eth-dispatch",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable putting dispatch on ethernet cores - place it on worker cores instead",
        )
        Perf.register_arg(
            name="--ignore-version",
            type=bool,
            default=False,
            choices=[True, False],
            help="Ignore check for Major/Minor/Patch between flatbuffer and TTRT, use at your own risk.",
        )
        Perf.register_arg(
            name="--enable-program-cache",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable program cache in ttnn runtime",
        )
        Perf.register_arg(
            name="--emitc",
            type=bool,
            default=False,
            choices=[True, False],
            help="toggles emitc testing",
        )
        Perf.register_arg(
            name="--trace-region-size",
            type=int,
            default=0,
            choices=None,
            help="Device trace region size",
        )
        Perf.register_arg(
            name="--dump-device-rate",
            type=int,
            default=1000,
            choices=None,
            help="Rate at which to flush device perf information",
        )
        Perf.register_arg(
            name="--benchmark",
            type=bool,
            default=False,
            choices=[True, False],
            help="Enable benchmark mode with warmup and e2e time measurements (automatically enables program cache)",
        )
        Perf.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        Perf.register_arg(
            name="--disable-ttrt-callbacks",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable ttrt callbacks",
        )
        Perf.register_arg(
            name="--filter",
            type=str,
            default="",
            choices=None,
            help="comma-separated list of operation types to filter out from perf results (e.g., 'const_eval,input_layout_conversion')",
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
            if not bin.check_version(ignore=self["--ignore-version"]):
                self.logger.warning(
                    "Flatbuffer version not present, are you sure that the binary is valid? - Skipped"
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
                    bin.check_version(ignore=self["--ignore-version"])
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.logging.warning(
                        f"SKIP: test={path} was skipped with exception={str(e)}"
                    )
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
                        self.logging.warning(
                            f"SKIP: test={path} was skipped with exception={message}"
                        )
                        self.results.add_result(test_result)
                        continue

                self.ttnn_binaries.append(bin)

            self.logging.debug(f"finished checking constraints for run API")

            for path in ttmetal_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                try:
                    bin.check_version(ignore=self["--ignore-version"])
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.logging.warning(
                        f"SKIP: test={path} was skipped with exception={str(e)}"
                    )
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
                        self.logging.warning(
                            f"SKIP: test={path} was skipped with exception={message}"
                        )
                        self.results.add_result(test_result)
                        continue

                self.ttmetal_binaries.append(bin)

        self.logging.debug(f"------finished checking constraints for perf API")

    def execute(self):
        self.logging.debug(f"------executing perf API")

        profiler_logs_dir = (
            f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs"
        )
        tracy_file_path = "tracy_profile_log_host.tracy"
        tracy_ops_times_file_path = "tracy_ops_times.csv"
        tracy_ops_data_file_path = "tracy_ops_data.csv"
        profiler_device_side_log_path = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/profile_log_device.csv"
        profiler_csv_file_path = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/reports/ops_perf_results.csv"

        self.file_manager.remove_directory(profiler_logs_dir)
        self.file_manager.create_directory(profiler_logs_dir)

        def _execute(binaries):
            # need to temporary add these sys paths so TTRT whls can find the `process_ops` function
            # ideally we want process_ops to be in a standalone module we can import from tt_metal
            sys.path.append(f"{get_ttrt_metal_home_path()}")
            sys.path.append(f"{get_ttrt_metal_home_path()}/ttnn")

            from tracy.process_ops_logs import process_ops

            def get_available_port():
                ip = socket.gethostbyname(socket.gethostname())
                self.logging.warning(f"Tracy binding to IP: {ip} (from gethostbyname(gethostname()))")

                for port in range(8086, 8500):
                    try:
                        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        serv.bind((ip, port))
                        self.logging.warning(f"Tracy selected port: {port}")
                        return str(port)
                    except PermissionError as e:
                        pass
                    except OSError as e:
                        pass
                return None

            if len(binaries) == 0:
                self.logging.warning(f"no binaries found to run - returning early")
                return

            for bin in binaries:
                try:
                    port = (
                        get_available_port() if self["--port"] == 0 else self["--port"]
                    )

                    if not port:
                        raise Exception("No available port found")
                    self.logging.debug(f"selected port={port}")

                    env_vars = dict(os.environ)
                    env_vars["TRACY_PORT"] = port

                    if not self["--host-only"]:
                        env_vars["TT_METAL_CLEAR_L1"] = "1"
                        env_vars["TT_METAL_DEVICE_PROFILER"] = "1"
                        env_vars["TTNN_OP_PROFILER"] = "1"
                        env_vars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

                    tracy_capture_tool_command = f"{self.tracy_capture_tool_path} -o {tracy_file_path} -f -p {port}"
                    self.tracy_capture_tool_process = subprocess.Popen(
                        tracy_capture_tool_command, shell=True
                    )

                    command_options = f"--enable-perf-trace --program-index {self['--program-index']} --loops {self['--loops']} --save-artifacts "

                    if self["--memory"]:
                        command_options += " --memory "

                    if self["--disable-eth-dispatch"]:
                        command_options += " --disable-eth-dispatch "

                    if self["--disable-golden"]:
                        command_options += " --disable-golden "

                    if self["--enable-program-cache"]:
                        command_options += " --enable-program-cache "

                    if self["--emitc"]:
                        command_options += " --emitc "

                    if self["--trace-region-size"] > 0:
                        command_options += (
                            f" --trace-region-size {self['--trace-region-size']} "
                        )

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

                    ttrt_executable_path = shutil.which("ttrt")
                    test_command = (
                        f"{ttrt_executable_path} run {bin.file_path} {command_options}"
                    )
                    self.logging.info(
                        f"test command for binary={bin.file_path} is: {test_command}"
                    )
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
                    except subprocess.TimeoutExpired as e:
                        self.tracy_capture_tool_process.terminate()
                        self.tracy_capture_tool_process.communicate()
                        raise Exception(
                            f"No profiling data could be captured. Please make sure you are on the correct build"
                        )

                    with open(tracy_ops_times_file_path, "w") as csv_file:
                        child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
                        child_calls_str = f"-x {','.join(child_calls)}"
                        subprocess.run(
                            f"{self.tracy_csvexport_tool_path} -u -p TT_DNN {child_calls_str} {tracy_file_path}",
                            shell=True,
                            check=True,
                            stdout=csv_file,
                            stderr=subprocess.DEVNULL,
                        )

                    self.logging.info(
                        f"host side ops time report generated at {tracy_ops_times_file_path}"
                    )

                    with open(tracy_ops_data_file_path, "w") as csv_file:
                        subprocess.run(
                            f'{self.tracy_csvexport_tool_path} -m -s ";" {tracy_file_path}',
                            shell=True,
                            check=True,
                            stdout=csv_file,
                            stderr=subprocess.DEVNULL,
                        )

                    self.logging.info(
                        f"host side ops data report generated at {tracy_ops_data_file_path}"
                    )

                    # copy all relevant files to correct folder directory (metal hardcoded path, need to make more dynamic from metal library)
                    self.file_manager.copy_file(profiler_logs_dir, tracy_file_path)
                    self.file_manager.copy_file(
                        profiler_logs_dir, tracy_ops_times_file_path
                    )
                    self.file_manager.copy_file(
                        profiler_logs_dir, tracy_ops_data_file_path
                    )

                    # copy all relevant files into perf folder for this test
                    perf_folder_path = self.artifacts.get_binary_perf_folder_path(bin)
                    self.file_manager.copy_file(perf_folder_path, tracy_file_path)
                    self.file_manager.copy_file(
                        perf_folder_path, tracy_ops_times_file_path
                    )
                    self.file_manager.copy_file(
                        perf_folder_path, tracy_ops_data_file_path
                    )

                    if not self["--host-only"]:
                        self.file_manager.copy_file(
                            perf_folder_path, profiler_device_side_log_path
                        )

                    process_ops(None, None, False)

                    # Add post-processing steps to insert location data into the ops_perf data file
                    # Get the op location to it's global call count mapping
                    def get_mlir_analysis_results(key):
                        call_count_mapping = {}

                        with open(tracy_ops_data_file_path, "r") as file:
                            lines = iter(file)
                            buffer = None
                            
                            # Debug: Log first 30 lines of the file (use WARNING level to show in CI)
                            self.logging.warning(f"DEBUG: Reading {tracy_ops_data_file_path} for key '{key}'")
                            file.seek(0)
                            first_lines = [file.readline().strip() for _ in range(30)]
                            for i, fline in enumerate(first_lines):
                                self.logging.warning(f"DEBUG: Line {i}: {fline[:150]}")  # Truncate long lines
                            file.seek(0)
                            lines = iter(file)

                            while True:
                                # Use buffered line if available, otherwise get next
                                line = buffer if buffer else next(lines, None)
                                buffer = None

                                if line is None:
                                    break  # End of file

                                # Find all the TT_DNN_DEVICE_OP under this LOC and record their global call counts
                                line = line.strip()
                                if key in line:
                                    # Format of line is
                                    # MLIR_OP_LOCATION;loc("/code/tt-mlir/build/test/ttmlir/Silicon/TTNN/n150/const-eval/Output/const-eval.mlir.tmp.mlir":17:14);5420869271
                                    # MLIR_CONST_EVAL_OP;true;6449925338
                                    parts = line.split(";")
                                    data = parts[1]
                                    self.logging.warning(f"DEBUG: Found {key} line, data='{data}'")
                                    block = []
                                    for next_line in lines:
                                        next_line = next_line.strip()
                                        if key in next_line:
                                            buffer = (
                                                next_line  # Save for next outer loop
                                            )
                                            break
                                        elif "TT_DNN_DEVICE_OP" in next_line:
                                            self.logging.warning(f"DEBUG: Found TT_DNN_DEVICE_OP line: {next_line[:100]}")
                                            block.append(next_line)

                                    # Process the collected block. Find it's global call count and add it to the loc
                                    self.logging.warning(f"DEBUG: Processing {len(block)} TT_DNN_DEVICE_OP lines for this LOC")
                                    for i, bline in enumerate(block):
                                        self.logging.warning(f"DEBUG: Block line {i}: {bline[:100]}")
                                        parts = bline.split(",")
                                        self.logging.warning(f"DEBUG: Split into {len(parts)} parts")
                                        if len(parts) > 3:
                                            # Strip and split part[3] on semicolon or space, and grab the number
                                            num_part = parts[3].strip()
                                            self.logging.warning(f"DEBUG: parts[3] = '{num_part}'")
                                            digits = ""
                                            for c in num_part:
                                                if c.isdigit():
                                                    digits += c
                                                else:
                                                    break
                                            global_call_count = (
                                                int(digits) if digits else None
                                            )
                                            self.logging.warning(f"DEBUG: Extracted global_call_count={global_call_count}, mapping to data='{data}'")
                                            call_count_mapping[global_call_count] = data
                                        else:
                                            self.logging.warning(f"DEBUG: ERROR - Not enough parts in bline!")
                                    
                                    self.logging.warning(f"DEBUG: After processing, call_count_mapping has {len(call_count_mapping)} entries")

                        return call_count_mapping

                    global_call_count_loc_mapping = get_mlir_analysis_results(
                        "MLIR_OP_LOCATION"
                    )
                    global_call_count_const_eval_op_mapping = get_mlir_analysis_results(
                        "MLIR_CONST_EVAL_OP"
                    )
                    global_call_count_input_layout_conversion_op_mapping = (
                        get_mlir_analysis_results("MLIR_INPUT_LAYOUT_CONVERSION_OP")
                    )
                    global_call_count_program_metadata_op_mapping = (
                        get_mlir_analysis_results("MLIR_PROGRAM_METADATA")
                    )

                    # Add location data, const_eval_op data, input_layout_conversion_op data and program metadata to profiler csv file
                    dir_name = os.path.dirname(profiler_csv_file_path)
                    base_name = os.path.basename(profiler_csv_file_path)
                    file_root, file_ext = os.path.splitext(base_name)
                    temp_file = os.path.join(dir_name, f"{file_root}_temp{file_ext}")

                    with open(
                        profiler_csv_file_path, mode="r", newline=""
                    ) as infile, open(temp_file, mode="w", newline="") as outfile:
                        reader = csv.DictReader(infile)
                        fieldnames = reader.fieldnames + [
                            "LOC",
                            "CONST_EVAL_OP",
                            "INPUT_LAYOUT_CONVERSION_OP",
                            "PROGRAM_METADATA",
                        ]
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for row in reader:
                            # Access the value at "GLOBAL CALL COUNT"
                            local_call_count = row.get("GLOBAL CALL COUNT")
                            local_call_count = int(local_call_count.strip())

                            # Append the location column with its location data
                            if local_call_count in global_call_count_loc_mapping.keys():
                                row["LOC"] = global_call_count_loc_mapping[
                                    local_call_count
                                ]
                            else:
                                row["LOC"] = 'loc("unknown")'

                            # Append the const_eval_op column with its const_eval_op data
                            if (
                                local_call_count
                                in global_call_count_const_eval_op_mapping.keys()
                            ):
                                row[
                                    "CONST_EVAL_OP"
                                ] = global_call_count_const_eval_op_mapping[
                                    local_call_count
                                ]
                            else:
                                row["CONST_EVAL_OP"] = "false"

                            # Append the input_layout_conversion_op column with its input_layout_conversion_op data
                            if (
                                local_call_count
                                in global_call_count_input_layout_conversion_op_mapping.keys()
                            ):
                                row[
                                    "INPUT_LAYOUT_CONVERSION_OP"
                                ] = global_call_count_input_layout_conversion_op_mapping[
                                    local_call_count
                                ]
                            else:
                                row["INPUT_LAYOUT_CONVERSION_OP"] = "false"

                            # Append the program metadata column with its metadata
                            if (
                                local_call_count
                                in global_call_count_program_metadata_op_mapping.keys()
                            ):
                                row[
                                    "PROGRAM_METADATA"
                                ] = global_call_count_program_metadata_op_mapping[
                                    local_call_count
                                ]
                            else:
                                row["PROGRAM_METADATA"] = "{}"

                            writer.writerow(row)

                    os.replace(temp_file, profiler_csv_file_path)

                    self.file_manager.copy_file(
                        perf_folder_path,
                        profiler_csv_file_path,
                    )

                    # Helper function to create filtered CSV files
                    def create_filtered_csv(filter_conditions, suffix):
                        """
                        Create a filtered CSV file based on filter conditions.

                        Args:
                            filter_conditions: dict mapping column names to expected values (e.g., {"CONST_EVAL_OP": "false"})
                            suffix: suffix for the output filename
                        """
                        dir_name = os.path.dirname(profiler_csv_file_path)
                        base_name = os.path.basename(profiler_csv_file_path)
                        file_root, file_ext = os.path.splitext(base_name)
                        filtered_file_path = os.path.join(
                            dir_name, f"{file_root}_{suffix}{file_ext}"
                        )

                        with open(
                            profiler_csv_file_path,
                            mode="r",
                            newline="",
                            encoding="utf-8",
                        ) as infile:
                            reader = csv.DictReader(infile)

                            with open(
                                filtered_file_path,
                                mode="w",
                                newline="",
                                encoding="utf-8",
                            ) as outfile:
                                writer = csv.DictWriter(
                                    outfile, fieldnames=reader.fieldnames
                                )
                                writer.writeheader()

                                for row in reader:
                                    # Check if all filter conditions are met
                                    include_row = True
                                    for (
                                        column,
                                        expected_value,
                                    ) in filter_conditions.items():
                                        actual_value = row.get(column, "").replace(
                                            " ", ""
                                        )
                                        if actual_value != expected_value:
                                            include_row = False
                                            break

                                    if include_row:
                                        writer.writerow(row)

                        self.file_manager.copy_file(
                            perf_folder_path, filtered_file_path
                        )
                        return filtered_file_path

                    # Parse the --filter argument to determine which filters to apply
                    filters_to_apply = []
                    if self["--filter"]:
                        filters_to_apply = [
                            f.strip().lower() for f in self["--filter"].split(",")
                        ]

                    # Always create the legacy const_eval filter for backward compatibility
                    create_filtered_csv({"CONST_EVAL_OP": "false"}, "minus_const_eval")

                    # Always create the input_layout_conversion filter
                    create_filtered_csv(
                        {"INPUT_LAYOUT_CONVERSION_OP": "false"},
                        "minus_input_layout_conversions",
                    )

                    # Create combined filter or user-specified filters
                    if not filters_to_apply:
                        # Default behavior: create combined filter excluding both const_eval and input_layout_conversions
                        create_filtered_csv(
                            {
                                "CONST_EVAL_OP": "false",
                                "INPUT_LAYOUT_CONVERSION_OP": "false",
                            },
                            "minus_const_eval_and_input_layout_conversions",
                        )
                    else:
                        # Create custom filters based on user input
                        filter_conditions = {}
                        filter_suffix_parts = []

                        for filter_type in filters_to_apply:
                            if filter_type == "const_eval":
                                filter_conditions["CONST_EVAL_OP"] = "false"
                                filter_suffix_parts.append("const_eval")
                            elif filter_type == "input_layout_conversion":
                                filter_conditions[
                                    "INPUT_LAYOUT_CONVERSION_OP"
                                ] = "false"
                                filter_suffix_parts.append("input_layout_conversions")
                            else:
                                self.logging.warning(
                                    f"Unknown filter type: {filter_type}. Supported types: const_eval, input_layout_conversion"
                                )

                        if filter_conditions:
                            suffix = "minus_" + "_and_".join(filter_suffix_parts)
                            create_filtered_csv(filter_conditions, suffix)

                    # post-process test results
                    test_result = []
                    with open("run_results.json", "r") as file:
                        test_result = json.load(file)

                    for result in test_result:
                        if result["result"] != "pass":
                            if result["result"] == "test_error":
                                raise TTRTTestException(str(result["exception"]))
                            raise Exception(f'{result["exception"]}')

                        if result["file_path"] == bin.file_path:
                            # post-process statistics for ttnn host duration, device fw duration
                            bin.program_results = result["program_results"]
                            total_ttnn_api_duration_map = {}
                            total_device_kernel_duration_map = {}

                            with open(
                                profiler_csv_file_path,
                                mode="r",
                                newline="",
                                encoding="utf-8",
                            ) as csvfile:
                                reader = csv.DictReader(csvfile)

                                for row in reader:
                                    const_eval_op = bool(row.get("CONST_EVAL_OP"))
                                    program_metadata = ast.literal_eval(
                                        row.get("PROGRAM_METADATA")
                                    )
                                    device_kernel_duration = int(
                                        row.get("DEVICE KERNEL DURATION [ns]")
                                    )
                                    ttnn_api_duration = int(
                                        row.get("HOST DURATION [ns]")
                                    )

                                    if len(program_metadata) == 0:
                                        continue

                                    program_index = program_metadata["program_index"]
                                    loop_number = program_metadata["loop_number"]

                                    if (
                                        program_index
                                        not in total_ttnn_api_duration_map.keys()
                                    ):
                                        total_ttnn_api_duration_map[program_index] = {}

                                    if (
                                        program_index
                                        not in total_device_kernel_duration_map.keys()
                                    ):
                                        total_device_kernel_duration_map[
                                            program_index
                                        ] = {}

                                    if (
                                        loop_number
                                        not in total_ttnn_api_duration_map[
                                            program_index
                                        ].keys()
                                    ):
                                        total_ttnn_api_duration_map[program_index][
                                            loop_number
                                        ] = 0

                                    if (
                                        loop_number
                                        not in total_device_kernel_duration_map[
                                            program_index
                                        ].keys()
                                    ):
                                        total_device_kernel_duration_map[program_index][
                                            loop_number
                                        ] = 0

                                    total_device_kernel_duration_map[program_index][
                                        loop_number
                                    ] += device_kernel_duration
                                    total_ttnn_api_duration_map[program_index][
                                        loop_number
                                    ] += ttnn_api_duration

                            for (
                                program_index,
                                loop_dic,
                            ) in total_ttnn_api_duration_map.items():
                                for loop_number, duration in loop_dic.items():
                                    bin.update_total_ttnn_api_duration_ns(
                                        program_index, loop_number, duration
                                    )

                            for (
                                program_index,
                                loop_dic,
                            ) in total_device_kernel_duration_map.items():
                                for loop_number, duration in loop_dic.items():
                                    bin.update_total_device_kernel_duration_ns(
                                        program_index, loop_number, duration
                                    )

                except Exception as e:
                    result = "error"
                    if isinstance(e, TTRTTestException):
                        result = "test_error"
                    test_result = {
                        "file_path": bin.file_path,
                        "result": result,
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.logging.error(
                        f"ERROR: test={bin.file_path} experienced an error with exception={str(e)}"
                    )
                    self.results.add_result(test_result)
                    bin.test_result = result
                    traceback.print_exc()
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

        for bin in self.ttnn_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                    "program_results": bin.program_results,
                }
                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={bin.file_path}")
            else:
                self.logging.error(f"ERROR: test case={bin.file_path}")

        for bin in self.ttmetal_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                    "program_results": bin.program_results,
                }
                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={bin.file_path}")
            else:
                self.logging.error(f"ERROR: test case={bin.file_path}")

        self.results.save_results(self["--result-file"])

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

        return self.results.get_result_code(), self.results.get_results()

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
