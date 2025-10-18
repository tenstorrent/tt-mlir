# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import importlib.machinery
import sys
import signal
import io
import subprocess
import time
import shutil
import atexit
import traceback
from pathlib import Path
import ast

from ttrt.common.util import *
from ttrt.common.query import Query
from ttrt.common.run import Run

# Add tt-alchemist file utils.py to path for EmitPy tests
TT_MLIR_HOME = Path(os.environ.get("TT_MLIR_HOME", os.getcwd())).resolve()
utils_path = os.path.join(TT_MLIR_HOME, "tools/tt-alchemist/templates/python/local")
sys.path.append(utils_path)
# Add ttnn python package location to path for EmitPy tests
sys.path.append(f"{get_ttrt_metal_home_path()}/ttnn")


class EmitPy:
    registered_args = {}

    @staticmethod
    def initialize_api():
        EmitPy.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        EmitPy.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        EmitPy.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        EmitPy.register_arg(
            name="--program-index",
            type=str,
            default="all",
            choices=["all"] + [str(i) for i in range(0, 5)],
            help="the program inside the fbb to run",
        )
        EmitPy.register_arg(
            name="--loops",
            type=int,
            default=1,
            choices=None,
            help="number of loops",
        )
        EmitPy.register_arg(
            name="--result-file",
            type=str,
            default="emitpy_results.json",
            choices=None,
            help="test file to save results to",
        )
        EmitPy.register_arg(
            name="--disable-golden",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable golden comparison for intermediate and output tensors",
        )
        EmitPy.register_arg(
            name="--flatbuffer",
            type=str,
            default="",
            choices=None,
            help="Provide a file or directory path for flatbuffer binary files to compare outputs to",
        )
        EmitPy.register_arg(
            name="--memory",
            type=bool,
            default=False,
            choices=[True, False],
            help="dump memory reports after every op execution",
        )
        EmitPy.register_arg(
            name="--disable-eth-dispatch",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable putting dispatch on ethernet cores - place it on worker cores instead",
        )
        EmitPy.register_arg(
            name="--ignore-version",
            type=bool,
            default=False,
            choices=[True, False],
            help="Ignore check for Major/Minor/Patch between flatbuffer and TTRT, use at your own risk.",
        )
        EmitPy.register_arg(
            name="--enable-program-cache",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable program cache in ttnn runtime",
        )
        EmitPy.register_arg(
            name="--dump-device-rate",
            type=int,
            default=1000,
            choices=None,
            help="Rate at which to flush device perf information",
        )
        EmitPy.register_arg(
            name="--benchmark",
            type=bool,
            default=False,
            choices=[True, False],
            help="Enable benchmark mode with warmup and e2e time measurements (automatically enables program cache)",
        )
        EmitPy.register_arg(
            name="dylib",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        EmitPy.register_arg(
            name="--disable-ttrt-callbacks",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable ttrt callbacks",
        )
        EmitPy.register_arg(
            name="--print-input-output-tensors",
            type=bool,
            default=False,
            choices=[True, False],
            help="print input and output tensors",
        )
        EmitPy.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in EmitPy.registered_args.items():
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
        self.emitpy_dylibs = []
        self.ttnn_binaries = {}
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing emitpy API")

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        if self["--save-artifacts"]:
            self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing emitpy API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for emitpy API")

        emitpy_dylib_paths = self.file_manager.find_emitpy_dylib_paths(self["dylib"])

        self.logging.debug(f"emitpy_dylib_paths={emitpy_dylib_paths}")

        for path in emitpy_dylib_paths:
            dylib = EmitPyDylib(self.logger, self.file_manager, path)
            self.emitpy_dylibs.append(dylib)
            if self["--flatbuffer"]:
                if os.path.isdir(self["--flatbuffer"]):
                    corresponding_ttnn_path = (
                        self.file_manager.find_corresponding_ttnn_in_directory(
                            path, self["--flatbuffer"]
                        )
                    )
                else:
                    corresponding_ttnn_path = self["--flatbuffer"]
                self.logging.debug(
                    f"Found ttnn file corresponding to .py dylib ={corresponding_ttnn_path}"
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

        self.logging.debug(f"------finished checking constraints for emitpy API")

    def execute(self):
        import ttrt.runtime
        import ttnn
        import importlib.util

        self.logging.debug(f"------executing emitpy API")

        if len(self.emitpy_dylibs) == 0:
            self.logging.warning(f"no EmitPy dylibs found to run - returning early")
            return

        if "--init" in sys.argv:
            self["--disable-golden"] = True

        # Workaround for issue #5205: Reorder dylibs so that ones with corresponding flatbuffers run first
        solo_dylibs = []
        paired_dylibs = []
        for dylib in self.emitpy_dylibs:
            if dylib in self.ttnn_binaries:
                paired_dylibs.append(dylib)
            else:
                solo_dylibs.append(dylib)
        self.emitpy_dylibs = paired_dylibs + solo_dylibs

        for dylib in self.emitpy_dylibs:
            self.logging.info(f"evaluating python file={dylib.file_path}")

            try:
                compare_to_ttnn = False
                if dylib in self.ttnn_binaries:
                    bin = self.ttnn_binaries[dylib]
                    compare_to_ttnn = True

                    command_options = f"--program-index {self['--program-index']} --loops {self['--loops']} --save-artifacts "

                    if self["--artifact-dir"]:
                        command_options += f" --artifact-dir {self['--artifact-dir']} "

                    if self["--result-file"]:
                        command_options += f" --result-file {self['--result-file']} "

                    if self["--memory"]:
                        command_options += " --memory "

                    if self["--disable-eth-dispatch"]:
                        command_options += " --disable-eth-dispatch "

                    if self["--disable-golden"]:
                        command_options += " --disable-golden "

                    if self["--enable-program-cache"]:
                        command_options += " --enable-program-cache "

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

                self.logging.debug(f"loading module from file: {dylib.file_path}")
                # Get the module name from the file path
                module_name = os.path.splitext(os.path.basename(dylib.file_path))[0]

                # Load the module from the file path
                spec = importlib.util.spec_from_file_location(
                    module_name, dylib.file_path
                )
                module = importlib.util.module_from_spec(spec)

                # Add the module to sys.modules so it can be imported by other modules
                sys.modules[module_name] = module

                # Execute the module
                spec.loader.exec_module(module)
                self.logging.debug(
                    f"module {module_name} loaded and executed successfully"
                )

                # Parse the AST to find function names
                with open(dylib.file_path, "r") as f:
                    source_code = f.read()

                tree = ast.parse(source_code)
                program_names = []
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name != "main"
                        and node.name[0:18] != "create_inputs_for_"
                    ):
                        program_names.append(node.name)

                self.logging.debug(f"Program names found: {program_names}")

                if self["--program-index"] != "all":
                    if len(program_names) > int(self["--program-index"]):
                        self.logging.warning(
                            f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        )
                        return

                if not compare_to_ttnn:
                    for program_index in range(len(program_names)):
                        if self["--program-index"] != "all" and program_index != int(
                            self["--program-index"]
                        ):
                            continue

                        self.logging.debug(
                            f"evaluating program={program_names[program_index]} for python file={dylib.file_path}"
                        )
                        create_program_inputs = (
                            "create_inputs_for_" + program_names[program_index]
                        )
                        create_inputs_func = getattr(module, create_program_inputs)
                        inputs = create_inputs_func()
                        self.logging.debug(f"created {len(inputs)} input tensors")

                        for loop in range(self["--loops"]):
                            self.logging.debug(
                                f"starting loop={loop+1}/{self['--loops']} for program={program_names[program_index]}"
                            )

                            # Save input tensors before they get deallocated
                            if self["--save-artifacts"]:
                                program_folder = f"{self.artifacts.get_dylib_emitpy_folder_path(dylib)}/program_{program_index}"
                                for i, input_tensor in enumerate(inputs):
                                    input_tensor = ttnn.from_device(input_tensor)
                                    torch_input = input_tensor.to_torch()

                                    self.artifacts.save_torch_tensor(
                                        program_folder,
                                        torch_input,
                                        f"emitpy_input_{i}.pt",
                                    )

                            if self["--print-input-output-tensors"]:
                                for i, input_tensor in enumerate(inputs):
                                    input_tensor = ttnn.from_device(input_tensor)
                                    torch_input = input_tensor.to_torch()
                                    self.logging.info(
                                        f"Input tensor {i}: {torch_input}"
                                    )

                            program_func = getattr(module, program_names[program_index])
                            dylib_outputs = program_func(inputs)
                            self.logging.debug(
                                f"finished loop={loop+1}/{self['--loops']} for program={program_names[program_index]}"
                            )

                            if self["--save-artifacts"]:
                                program_folder = f"{self.artifacts.get_dylib_emitpy_folder_path(dylib)}/program_{program_index}"
                                for i, output in enumerate(dylib_outputs):
                                    ttnn_output = ttnn.from_device(output)
                                    torch_output = ttnn_output.to_torch()

                                    self.artifacts.save_torch_tensor(
                                        program_folder,
                                        torch_output,
                                        f"emitpy_output_{i}.pt",
                                    )

                            if self["--print-input-output-tensors"]:
                                for i, output_tensor in enumerate(dylib_outputs):
                                    output_tensor = ttnn.from_device(output_tensor)
                                    torch_output = output_tensor.to_torch()
                                    self.logging.info(
                                        f"Output tensor {i}: {torch_output}"
                                    )
                else:
                    with ttnn.manage_device(device_id=0) as device:
                        for program_index in range(len(program_names)):
                            if self[
                                "--program-index"
                            ] != "all" and program_index != int(
                                self["--program-index"]
                            ):
                                continue
                            self.logging.debug(
                                f"evaluating program={program_names[program_index]} for python file={dylib.file_path}"
                            )

                            self.logging.debug(
                                f"loading input tensors from artifacts for program={program_index}"
                            )
                            torch_inputs = self.load_tensors_from_artifacts(
                                bin, "input"
                            )["program_" + str(program_index)]
                            inputs = []
                            for i in torch_inputs:
                                inputs.append(
                                    ttnn.as_tensor(
                                        i,
                                        layout=ttnn.Layout.TILE,
                                        device=device,
                                        memory_config=ttnn.MemoryConfig(
                                            ttnn.TensorMemoryLayout.INTERLEAVED,
                                            ttnn.BufferType.DRAM,
                                            None,
                                        ),
                                    )
                                )

                            # Save artifacts before they get deallocated
                            if self["--save-artifacts"]:
                                program_folder = f"{self.artifacts.get_dylib_emitpy_folder_path(dylib)}/program_{program_index}"
                                for i, input_tensor in enumerate(torch_inputs):
                                    self.artifacts.save_torch_tensor(
                                        program_folder,
                                        input_tensor,
                                        f"emitpy_input_{i}.pt",
                                    )

                            if self["--print-input-output-tensors"]:
                                for i, input_tensor in enumerate(inputs):
                                    input_tensor = ttnn.from_device(input_tensor)
                                    torch_input = input_tensor.to_torch()
                                    self.logging.info(
                                        f"Input tensor {i}: {torch_input}"
                                    )

                            for loop in range(self["--loops"]):
                                self.logging.debug(
                                    f"starting loop={loop+1}/{self['--loops']} for program={program_names[program_index]}"
                                )
                                program_func = getattr(
                                    module, program_names[program_index]
                                )
                                dylib_outputs = program_func(inputs)
                                self.logging.debug(
                                    f"finished loop={loop+1}/{self['--loops']} for program={program_names[program_index]}"
                                )

                                self.logging.debug(
                                    f"comparing flatbuffer outputs to emitpy outputs"
                                )
                                torch_dylib_outputs = []
                                for output in dylib_outputs:
                                    output = ttnn.from_device(output)
                                    torch_dylib_outputs.append(output.to_torch())

                                torch_fbb_outputs = self.load_tensors_from_artifacts(
                                    bin, "device_output"
                                )["program_" + str(program_index)]

                                if self["--save-artifacts"]:
                                    program_folder = f"{self.artifacts.get_dylib_emitpy_folder_path(dylib)}/program_{program_index}"
                                    for i, output in enumerate(torch_dylib_outputs):
                                        self.artifacts.save_torch_tensor(
                                            program_folder,
                                            output,
                                            f"emitpy_output_{i}.pt",
                                        )

                                if self["--print-input-output-tensors"]:
                                    for i, output_tensor in enumerate(dylib_outputs):
                                        output_tensor = ttnn.from_device(output_tensor)
                                        torch_output = output_tensor.to_torch()
                                        self.logging.info(
                                            f"Output tensor {i}: {torch_output}"
                                        )

                                # Compare outputs
                                for i in range(len(torch_fbb_outputs)):
                                    # Nan and Inf handling
                                    torch_dylib_outputs[i] = mask_torch_inf_nan(
                                        torch_dylib_outputs[i]
                                    )
                                    torch_fbb_outputs[i] = mask_torch_inf_nan(
                                        torch_fbb_outputs[i]
                                    )

                                    if not torch.allclose(
                                        torch_dylib_outputs[i], torch_fbb_outputs[i]
                                    ):
                                        self.logging.error(
                                            f"EmitPy dylib output tensor does not match flatbuffer output for program_index={program_index}, loop={loop}"
                                        )
                                        self.logging.debug(
                                            f"EmitPy dylib output tensor {torch_dylib_outputs[i]}"
                                        )
                                        self.logging.debug(
                                            f"Flatbuffer output tensor {torch_fbb_outputs[i]}"
                                        )
                                        raise Exception(
                                            f"EmitPy dylib output tensor does not match flatbuffer output for program_index={program_index}, loop={loop}"
                                        )
                                    else:
                                        self.logging.debug(
                                            f"Output tensors match for program_index={program_index}, loop={loop}"
                                        )
                                self.logging.info(
                                    f"All output tensors match for {dylib.file_path}"
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

        self.logging.debug(f"------finished executing emitpy API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing emitpy API")

        for dylib in self.emitpy_dylibs:
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

        self.logging.debug(f"------finished postprocessing emitpy API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting emitpy API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished emitpy API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

    @staticmethod
    def register_arg(name, type, default, choices, help):
        EmitPy.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        emitpy_parser = subparsers.add_parser(
            "emitpy",
            help="run EmitPy Dylib tests and optionally compare outputs to TTNN",
        )
        emitpy_parser.set_defaults(api=EmitPy)

        for name, attributes in EmitPy.registered_args.items():
            if name == "dylib":
                emitpy_parser.add_argument(f"{name}", help=attributes["help"])
            elif attributes["type"] == bool:
                emitpy_parser.add_argument(
                    f"{name}",
                    action="store_true",
                    help=attributes["help"],
                )
            else:
                emitpy_parser.add_argument(
                    f"{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=attributes["choices"],
                    help=attributes["help"],
                )

        return emitpy_parser

    def load_tensors_from_artifacts(self, bin, key):
        """
        Open directory, loop through subdirectories, load all .pt files into torch tensors, and save them according to their respective program.
        """
        fbb_run_directory = self.artifacts.get_binary_run_folder_path(bin)
        program_tensors = {}
        program_names = [d for d in os.listdir(fbb_run_directory)]
        self.logging.debug(f"Loading .pt tensors from directory: {fbb_run_directory}")
        for program in program_names:
            program_dir = os.path.join(fbb_run_directory, program)
            files = sorted([d for d in os.listdir(program_dir)])
            tensors = []
            for file in files:
                file = os.path.join(program_dir, file)
                if file.endswith(".pt") and key in file:
                    try:
                        tensors.append(torch.load(file, weights_only=True))
                        self.logging.debug(f"Loading tensor from file: {file}")
                    except Exception as e:
                        raise Exception(
                            f"Error loading tensor from file {file}: {str(e)}"
                        )

            program_tensors[program] = tensors

        return program_tensors
