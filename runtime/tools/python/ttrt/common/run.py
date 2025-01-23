# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from ttrt.common.util import *
from ttrt.common.query import Query
from ttrt.common.callback import get_callback_fn, CallbackRuntimeConfig


class Run:
    registered_args = {}

    @staticmethod
    def initialize_api():
        Run.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        Run.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        Run.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        Run.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        Run.register_arg(
            name="--program-index",
            type=str,
            default="all",
            choices=["all"] + [str(i) for i in range(0, 5)],
            help="the program inside the fbb to run",
        )
        Run.register_arg(
            name="--loops",
            type=int,
            default=1,
            choices=None,
            help="number of loops",
        )
        Run.register_arg(
            name="--init",
            type=str,
            default="randn",
            choices=Run.TorchInitializer.init_fns,
            help="function to initialize tensors with",
        )
        Run.register_arg(
            name="--identity",
            type=bool,
            default=False,
            choices=[True, False],
            help="do a golden identity test on the output tensors",
        )
        Run.register_arg(
            name="--non-zero",
            type=bool,
            default=False,
            choices=[True, False],
            help="test the output tensors are non-zero",
        )
        Run.register_arg(
            name="--rtol",
            type=float,
            default=1e-05,
            choices=None,
            help="rtol for golden test",
        )
        Run.register_arg(
            name="--atol",
            type=float,
            default=1e-08,
            choices=None,
            help="atol for golden test",
        )
        Run.register_arg(
            name="--pcc",
            type=float,
            default=0.99,
            choices=None,
            help="pcc for golden test",
        )
        Run.register_arg(
            name="--seed",
            type=int,
            default=0,
            choices=None,
            help="seed for random number generator",
        )
        Run.register_arg(
            name="--load-kernels-from-disk",
            type=bool,
            default=False,
            choices=[True, False],
            help="pickup the kernels from disk (/tmp) instead of the flatbuffer",
        )
        Run.register_arg(
            name="--enable-async-ttnn",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable async mode device execution for TTNN runtime",
        )
        Run.register_arg(
            name="--disable-maxpool2d-preshard",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable maxpool2d preshard workaround",
        )
        Run.register_arg(
            name="--disable-swap-binary-operands",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable swap binary operands workaround",
        )
        Run.register_arg(
            name="--disable-read-update-index-for-kv-cache",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable read update index for kv cache workaround",
        )
        Run.register_arg(
            name="--disable-to-dtype-on-host",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable to_dtype on host workaround",
        )
        Run.register_arg(
            name="--result-file",
            type=str,
            default="run_results.json",
            choices=None,
            help="test file to save results to",
        )
        Run.register_arg(
            name="--emitc",
            type=bool,
            default=False,
            choices=[True, False],
            help="toggles emitc testing",
        )
        Run.register_arg(
            name="--disable-golden",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable golden comparison for intermediate and output tensors",
        )
        Run.register_arg(
            name="--save-golden-tensors",
            type=bool,
            default=False,
            choices=[True, False],
            help="save golden and device tensors that are compared during callback runtime",
        )
        Run.register_arg(
            name="--debugger",
            type=bool,
            default=False,
            choices=[True, False],
            help="run step debugger after every op execution",
        )
        Run.register_arg(
            name="--memory",
            type=bool,
            default=False,
            choices=[True, False],
            help="dump memory reports after every op execution (use in conjunction with --save-artifacts)",
        )
        Run.register_arg(
            name="--check-memory-leak",
            type=bool,
            default=False,
            choices=[True, False],
            help="check for memory leaks (use in conjunction with --memory)",
        )
        Run.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in Run.registered_args.items():
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
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing run API")
        self.query()

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        if self["--save-artifacts"]:
            self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing read API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for run API")

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
                self.logging.warning(
                    f"SKIP: test={path} was skipped with exception={str(e)}"
                )
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
                self.logging.warning(
                    f"SKIP: test={path} was skipped with exception={str(e)}"
                )
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

        self.logging.debug(f"------finished checking constraints for run API")

    def execute(self):
        self.logging.debug(f"------executing run API")

        def _execute(binaries):
            import ttrt.runtime
            import torch

            if len(binaries) == 0:
                self.logging.warning(f"no binaries found to run - returning early")
                return

            debug_env = ttrt.runtime.DebugEnv.get(
                self["--load-kernels-from-disk"], self["--enable-async-ttnn"]
            )
            self.logging.debug(f"setting tt runtime debug env={debug_env}")
            workaround_env = ttrt.runtime.WorkaroundEnv.get(
                not self["--disable-maxpool2d-preshard"],
                not self["--disable-swap-binary-operands"],
                not self["--disable-read-update-index-for-kv-cache"],
                not self["--disable-to-dtype-on-host"],
            )
            self.logging.debug(f"setting tt runtime workaround env={workaround_env}")
            self.logging.debug(f"setting torch manual seed={self['--seed']}")
            torch.manual_seed(self["--seed"])
            ttrt.runtime.set_compatible_runtime(binaries[0].fbb)
            current_runtime = ttrt.runtime.get_current_runtime()
            self.logging.debug(f"opening devices={self.query.device_ids}")
            device = ttrt.runtime.open_device(self.query.device_ids)

            callback_runtime_config = CallbackRuntimeConfig(
                device,
                "",
                self["--pcc"],
                self["--atol"],
                self["--rtol"],
                self["--save-golden-tensors"],
                self.logging,
                not self["--disable-golden"],
                self["--memory"],
                self["--debugger"],
            )

            callback_env = ttrt.runtime.DebugHooks.get(
                get_callback_fn(callback_runtime_config)
            )

            try:
                for bin in binaries:
                    try:
                        self.logging.info(f"evaluating binary={bin.file_path}")

                        if self["--save-artifacts"]:
                            self.artifacts.create_binary_artifacts_folder(bin)

                        if self["--emitc"]:
                            # .so are compiled such that they have the same name as flatbuffers, so we rename here
                            emitc_dylib_path = bin.file_path.replace(".ttnn", ".so")

                            # Open the dylib
                            emitc_dylib_handle = ttrt.runtime.testing.open_so(
                                emitc_dylib_path
                            )
                            self.logging.debug(f"opened emitc dylib={emitc_dylib_path}")

                        program_indices = []
                        if self["--program-index"] == "all":
                            program_indices.extend(range(bin.get_num_programs()))
                        else:
                            program_indices.append(int(self["--program-index"]))

                        for program_index in program_indices:
                            self.logging.debug(
                                f"evaluating program={program_index} for binary={bin.file_path}"
                            )

                            callback_runtime_config.start_new_callback(
                                f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}"
                            )

                            program = bin.get_program(program_index)
                            golden_inputs = []

                            for i in range(len(program.program["inputs"])):
                                golden_tensor = None

                                self.logging.info("tapssssss")

                                if not self["--disable-golden"]:
                                    golden_tensor = bin.fbb.get_debug_info_golden(
                                        f"input_{i}"
                                    )

                                self.logging.info("tapssssss")
                                #breakpoint()

                                if golden_tensor is not None:

                                    dtype = ttrt_datatype_to_torch_dtype(
                                        golden_tensor.dtype
                                    )

                                    golden_tensor_torch = torch.frombuffer(
                                        golden_tensor, dtype=dtype
                                    )
                                    golden_inputs.append(golden_tensor_torch)

                                    self.logging.info("tapssssss")
                                    self.logging.info(golden_tensor_torch)
                                    self.logging.info("tapssssss")

                            program.populate_inputs(
                                Run.TorchInitializer.get_initilizer(self["--init"]),
                                golden_inputs,
                            )
                            program.populate_outputs(
                                Run.TorchInitializer.get_initilizer("zeros")
                            )

                            total_inputs = []
                            total_outputs = []
                            for loop in range(self["--loops"]):
                                self.logging.debug(
                                    f"generating inputs/outputs for loop={loop+1}/{self['--loops']} for binary={bin.file_path}"
                                )

                                inputs = []
                                outputs = []
                                for i in program.input_tensors:
                                    inputs.append(
                                        ttrt.runtime.create_tensor(
                                            i.data_ptr(),
                                            list(i.shape),
                                            list(i.stride()),
                                            i.element_size(),
                                            Binary.Program.to_data_type(i.dtype),
                                        )
                                    )

                                for i in program.output_tensors:
                                    outputs.append(
                                        ttrt.runtime.create_tensor(
                                            i.data_ptr(),
                                            list(i.shape),
                                            list(i.stride()),
                                            i.element_size(),
                                            Binary.Program.to_data_type(i.dtype),
                                        )
                                    )

                                total_inputs.append(inputs)
                                total_outputs.append(outputs)

                            event = None

                            for loop in range(self["--loops"]):
                                self.logging.debug(
                                    f"starting loop={loop+1}/{self['--loops']} for binary={bin.file_path}"
                                )
                                if (
                                    current_runtime
                                    == ttrt.runtime.DeviceRuntime.TTMetal
                                ):
                                    event = ttrt.runtime.submit(
                                        device,
                                        bin.fbb,
                                        program_index,
                                        total_inputs[loop],
                                        total_outputs[loop],
                                    )

                                elif current_runtime == ttrt.runtime.DeviceRuntime.TTNN:
                                    runtime_outputs = ttrt.runtime.submit(
                                        device,
                                        bin.fbb,
                                        program_index,
                                        total_inputs[loop],
                                    )
                                    ttrt.runtime.wait(runtime_outputs)
                                    for i, runtime_output_tensor in enumerate(
                                        runtime_outputs
                                    ):
                                        ttrt.runtime.memcpy(
                                            total_outputs[loop][i],
                                            runtime_output_tensor,
                                        )
                                        ttrt.runtime.deallocate_tensor(
                                            runtime_output_tensor, force=True
                                        )

                                self.logging.debug(
                                    f"finished loop={loop+1}/{self['--loops']} for binary={bin.file_path}"
                                )

                            if event is not None:
                                ttrt.runtime.wait(event)

                            # Compare to EmitC
                            if self["--emitc"]:
                                # Create symbol string to read from dylib
                                fwd_func_name = program.program["name"]
                                fwd_func_name_len = len(fwd_func_name)
                                fwd_func_sym = f"_Z{fwd_func_name_len}{fwd_func_name}St6vectorIN2tt8tt_metal6TensorESaIS2_EEPNS1_2v07IDeviceE"

                                for loop in range(self["--loops"]):
                                    emitc_outs = ttrt.runtime.testing.run_so_program(
                                        emitc_dylib_handle,
                                        fwd_func_sym,
                                        total_inputs[loop],
                                        device,
                                    )
                                    self.logging.debug(
                                        f"got emitc outputs for program_index={program_index}, loop={loop}"
                                    )

                                    all_tensors_match = (
                                        ttrt.runtime.testing.compare_outs(
                                            total_outputs[0], emitc_outs
                                        )
                                    )

                                    if not all_tensors_match:
                                        self.logging.error(
                                            "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                        )
                                        self.logging.error(
                                            total_outputs[loop], emitc_outs
                                        )
                                        raise Exception(
                                            "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                        )

                            if self["--identity"]:
                                self.logging.debug(
                                    f"checking identity with rtol={self['--rtol']} and atol={self['--atol']}"
                                )

                                for i, o in zip(
                                    program.input_tensors, program.output_tensors
                                ):
                                    if not torch.allclose(
                                        i, o, rtol=self["--rtol"], atol=self["--atol"]
                                    ):
                                        self.logging.error(
                                            f"Failed: inputs and outputs do not match in binary"
                                        )
                                        self.logging.error(i - o)

                            if self["--non-zero"]:
                                self.logging.debug("checking outputs are non-zero")
                                for o in program.output_tensors:
                                    if not torch.any(o):
                                        self.logging.error(
                                            "Failed: output tensor all zero"
                                        )

                            self.logging.debug(
                                f"input tensors for program={program_index}"
                            )
                            for tensor in program.input_tensors:
                                self.logging.debug(f"{tensor}\n")

                            self.logging.debug(
                                f"output tensors for program={program_index}"
                            )
                            for tensor in program.output_tensors:
                                self.logging.debug(f"{tensor}\n")

                            device.deallocate_buffers()

                            # if golden comparison is enabled, check golden results json file to see if test passed
                            if not self["--disable-golden"]:
                                if self["--save-artifacts"]:
                                    callback_runtime_config.save_golden_report(
                                        f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}/golden_results.json"
                                    )

                                callback_runtime_config.check_pcc()

                            if self["--memory"]:
                                if self["--save-artifacts"]:
                                    callback_runtime_config.save_memory_report(
                                        f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}/memory_results.json"
                                    )

                                if self["--check-memory-leak"]:
                                    callback_runtime_config.check_memory_leak()

                    except Exception as e:
                        test_result = {
                            "file_path": bin.file_path,
                            "result": "error",
                            "exception": str(e),
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "program_index": self["--program-index"],
                        }
                        self.logging.error(
                            f"ERROR: test={bin.file_path} experienced an error with exception={str(e)}"
                        )
                        self.results.add_result(test_result)
                        bin.test_result = "error"
                        continue
            finally:
                ttrt.runtime.close_device(device)

        self.logging.debug(f"executing ttnn binaries")
        _execute(self.ttnn_binaries)
        self.logging.debug(f"finished executing ttnn binaries")

        self.logging.debug(f"executing ttmetal binaries")
        _execute(self.ttmetal_binaries)
        self.logging.debug(f"finished executing ttmetal binaries")

        self.logging.debug(f"------finished executing run API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing run API")

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
                }
                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={bin.file_path}")
            else:
                self.logging.error(f"ERROR: test case={bin.file_path}")

        self.results.save_results(self["--result-file"])

        self.logging.debug(f"------finished postprocessing run API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting run API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished run API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

    @staticmethod
    def register_arg(name, type, default, choices, help):
        Run.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        run_parser = subparsers.add_parser("run", help="run a flatbuffer binary")
        run_parser.set_defaults(api=Run)

        for name, attributes in Run.registered_args.items():
            if name == "binary":
                run_parser.add_argument(f"{name}", help=attributes["help"])
            elif attributes["type"] == bool:
                run_parser.add_argument(
                    f"{name}",
                    action="store_true",
                    help=attributes["help"],
                )
            else:
                run_parser.add_argument(
                    f"{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=attributes["choices"],
                    help=attributes["help"],
                )
        return run_parser

    class TorchInitializer:
        init_fns = sorted(["randn", "arange", "zeros", "ones"])

        @staticmethod
        def get_initilizer(name):
            for attr, value in Run.TorchInitializer.__dict__.items():
                if attr == name:
                    return value

            raise Exception(f"could not find specified init function={name}")

        @staticmethod
        def get_init_fns():
            return Run.TorchInitializer.init_fns

        @staticmethod
        def randn(shape, dtype):
            import torch

            if dtype in (torch.uint8, torch.uint16, torch.uint32):
                high = torch.iinfo(dtype).max + 1
                return torch.randint(0, high, shape, dtype=dtype)

            return torch.randn(shape, dtype=dtype)

        @staticmethod
        def arange(shape, dtype):
            import torch

            def volume(shape):
                v = 1
                for i in shape:
                    v *= i
                return v

            return torch.arange(volume(shape), dtype=dtype).reshape(shape)

        @staticmethod
        def zeros(shape, dtype):
            import torch

            return torch.zeros(shape, dtype=dtype)

        @staticmethod
        def ones(shape, dtype):
            import torch

            return torch.ones(shape, dtype=dtype)
