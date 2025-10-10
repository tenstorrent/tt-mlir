# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from functools import reduce
import operator

from ttrt.common.util import *
from ttrt.common.query import Query
from ttrt.common.callback import (
    pre_op_get_callback_fn,
    post_op_get_callback_fn,
    CallbackRuntimeConfig,
)


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
            help="function to initialize tensors with (implies --disable-golden)",
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
            help="rtol for the PCC and identity golden tests",
        )
        Run.register_arg(
            name="--atol",
            type=float,
            default=1e-08,
            choices=None,
            help="atol for the PCC and identity golden tests",
        )
        Run.register_arg(
            name="--rtol-allclose",
            type=float,
            default=5e-02,
            choices=None,
            help="rtol for the allclose golden test",
        )
        Run.register_arg(
            name="--atol-allclose",
            type=float,
            default=1e-03,
            choices=None,
            help="atol for the allclose golden test",
        )
        Run.register_arg(
            name="--pcc",
            type=float,
            default=0.99,
            choices=None,
            help="pcc for golden test",
        )
        Run.register_arg(
            name="--golden-diff-topk",
            type=int,
            default=10,
            choices=None,
            help="print the top k golden and output tensor elemtent pairs sorted by absolute/relative difference",
        )
        Run.register_arg(
            name="--seed",
            type=int,
            default=0,
            choices=None,
            help="seed for random number generator",
        )
        Run.register_arg(
            name="--dump-kernels-to-disk",
            type=bool,
            default=False,
            choices=[True, False],
            help="dump the kernels to disk (/tmp) as they are being executed",
        )
        Run.register_arg(
            name="--load-kernels-from-disk",
            type=bool,
            default=False,
            choices=[True, False],
            help="pickup the kernels from disk (/tmp) instead of the flatbuffer, must have previously run with --dump-kernels-to-disk",
        )
        Run.register_arg(
            name="--use-loc-for-kernel-name",
            type=bool,
            default=False,
            choices=[True, False],
            help="Use the location information to derive the kernel's filename when dumping to disk",
        )
        Run.register_arg(
            name="--kernel-source-dir",
            type=str,
            default="",
            choices=None,
            help="test dir to save kernels to",
        )
        Run.register_arg(
            name="--disable-device-address-validation",
            type=bool,
            default=False,
            choices=[True, False],
            help="validate device addresses are in legal ranges",
        )
        Run.register_arg(
            name="--blocking-cq",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable blocking CQ mode device execution",
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
            name="--disable-trace-implicit-from-device",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable trace from implicitly bouncing tensors off of host",
        )
        Run.register_arg(
            name="--disable-blackhole-workarounds",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable blackhole workarounds",
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
            name="--print-input-output-tensors",
            type=bool,
            default=False,
            choices=[True, False],
            help="print input and output tensors",
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
            name="--disable-eth-dispatch",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable putting dispatch on ethernet cores - place it on worker cores instead; necessary on blackhole",
        )
        Run.register_arg(
            name="--ignore-version",
            type=bool,
            default=False,
            choices=[True, False],
            help="Ignore check for Major/Minor/Patch between flatbuffer and TTRT, use at your own risk.",
        )
        Run.register_arg(
            name="--dirty-tensor-schedule",
            type=str,
            default="",
            choices=None,
            help="Configuration for dirtying tensors, format: 'index:iterations,...' (e.g., '0:1,2:3' to dirty tensor 0 after 1 iteration and tensor 2 after 3 iterations)",
        )
        Run.register_arg(
            name="--enable-program-cache",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable program cache in ttnn runtime",
        )
        Run.register_arg(
            name="--trace-region-size",
            type=int,
            default=0,
            choices=None,
            help="Device trace region size",
        )
        Run.register_arg(
            name="--dump-device-rate",
            type=int,
            default=1000,
            choices=None,
            help="Rate at which to flush device perf information",
        )
        Run.register_arg(
            name="--enable-perf-trace",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable performance tracing",
        )
        Run.register_arg(
            name="--benchmark",
            type=bool,
            default=False,
            choices=[True, False],
            help="Enable benchmark mode with warmup and e2e time measurements (automatically enables program cache)",
        )
        Run.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        Run.register_arg(
            name="--ones-density",
            type=int,
            default=1,
            choices=None,
            help="Random ones vs zeroes density, 1 = 100%% ones, 2 = 50%% ones, 3 = 33%% ones, etc.",
        )
        Run.register_arg(
            name="--fabric-config",
            type=str,
            default="fabric_1d",
            choices=None,
            help="Select fabric topology: disabled, fabric_1d, fabric_1d_ring, fabric_2d, fabric_2d_torus, fabric_2d_dynamic or custom (case-insensitive, default: disabled)",
        )
        Run.register_arg(
            name="--disable-ttrt-callbacks",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable ttrt callbacks",
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
        self.query = Query(
            {"--quiet": True, "--disable-eth-dispatch": self["--disable-eth-dispatch"]},
            self.logger,
            self.artifacts,
        )
        self.ttnn_binaries = []
        self.ttmetal_binaries = []
        self.cuda_binaries = []
        self.results = Results(self.logger, self.file_manager)
        self.torch_initializer = Run.TorchInitializer(self)

        # If benchmark mode is enabled, set certain defaults
        if self["--benchmark"]:
            self["--loops"] = 2
            self["--enable-program-cache"] = True
            self["--disable-golden"] = True
            self["--program-index"] = 0

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
        cuda_binary_paths = self.file_manager.find_cuda_binary_paths(self["binary"])

        self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")
        self.logging.debug(f"ttmetal_binary_paths={ttmetal_binary_paths}")
        self.logging.debug(f"cuda_binary_paths={cuda_binary_paths}")

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

            try:
                bin.check_system_desc(self.query, ignore=self["--ignore-version"])
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

            try:
                bin.check_system_desc(self.query, ignore=self["--ignore-version"])
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

        for path in cuda_binary_paths:
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

            self.cuda_binaries.append(bin)

        self.logging.debug(f"------finished checking constraints for run API")

    def execute(self):
        self.logging.debug(f"------executing run API")

        def _execute(binaries):
            import ttrt.runtime
            import torch

            def convert_input_layouts(device, inputs, fbb, program_index):
                import ttrt.runtime

                inputs_converted = []
                for input_index in range(len(inputs)):
                    input_layout = ttrt.runtime.get_layout(
                        fbb, program_index, input_index
                    )
                    perf_env.tracy_log_op_location(f"loc(arg_{input_index})")
                    inputs_converted.append(
                        ttrt.runtime.to_layout(
                            inputs[input_index], device, input_layout, True
                        )
                    )
                return inputs_converted

            # Create 'owned tensor' in case of empty tensor;
            # otherwise create 'borrowed tensor'.
            def create_tensor(tensor):
                # Empty tensor if any of the dim is zero.
                isEmptyTensor = not all(tensor.shape)

                if isEmptyTensor:
                    return ttrt.runtime.create_owned_host_tensor(
                        tensor.data_ptr(),
                        list(tensor.shape),
                        list(tensor.stride()),
                        tensor.element_size(),
                        Binary.Program.to_data_type(tensor.dtype),
                    )

                return ttrt.runtime.create_borrowed_host_tensor(
                    tensor.data_ptr(),
                    list(tensor.shape),
                    list(tensor.stride()),
                    tensor.element_size(),
                    Binary.Program.to_data_type(tensor.dtype),
                )

            if len(binaries) == 0:
                self.logging.warning(f"no binaries found to run - returning early")
                return

            debug_env = ttrt.runtime.DebugEnv.get(
                self["--dump-kernels-to-disk"],
                self["--load-kernels-from-disk"],
                self["--use-loc-for-kernel-name"],
                self["--kernel-source-dir"],
                not self["--disable-device-address-validation"],
                self["--blocking-cq"],
            )
            self.logging.debug(f"setting tt runtime debug env={debug_env}")
            workaround_env = ttrt.runtime.WorkaroundEnv.get(
                not self["--disable-swap-binary-operands"],
                not self["--disable-read-update-index-for-kv-cache"],
                not self["--disable-trace-implicit-from-device"],
                not self["--disable-blackhole-workarounds"],
            )
            self.logging.debug(f"setting tt runtime workaround env={workaround_env}")
            tracy_program_metadata = {
                "disable_eth_dispatch": self["--disable-eth-dispatch"],
                "enable_program_cache": self["--enable-program-cache"],
                "dump_device_rate": self["--dump-device-rate"],
            }
            perf_env = ttrt.runtime.PerfEnv.get(
                self["--dump-device-rate"],
                self["--enable-perf-trace"],
                str(tracy_program_metadata),
            )
            self.logging.debug(f"setting tt runtime perf env={perf_env}")
            self.logging.debug(f"setting torch manual seed={self['--seed']}")
            torch.manual_seed(self["--seed"])
            ttrt.runtime.set_compatible_device_runtime(binaries[0].fbb)
            current_runtime = ttrt.runtime.get_current_device_runtime()
            self.logging.debug(f"opening devices={self.query.device_ids}")
            dispatch_core_type = ttrt.runtime.DispatchCoreType.ETH

            if self["--disable-eth-dispatch"]:
                dispatch_core_type = ttrt.runtime.DispatchCoreType.WORKER

            if "--init" in sys.argv:
                self["--disable-golden"] = True

            num_devices = len(self.query.device_ids)
            mesh_options = ttrt.runtime.MeshDeviceOptions()
            mesh_options.dispatch_core_type = dispatch_core_type
            mesh_options.enable_program_cache = self["--enable-program-cache"]
            mesh_options.trace_region_size = self["--trace-region-size"]

            # Initialize `device` to `None` for error handling in case device opening fails
            device = None

            for bin in binaries:

                try:

                    fb_mesh_shape = bin.get_program(0).mesh_shape
                    num_mesh_devices = reduce(operator.mul, fb_mesh_shape, 1)
                    mesh_options.mesh_shape = fb_mesh_shape

                    # Verify that the expected number of devices in the fb mesh shape is valid on this system
                    if num_mesh_devices > num_devices:
                        raise Exception(
                            f"Not enough devices ({num_devices}) to run program with mesh shape {fb_mesh_shape}"
                        )

                    if num_mesh_devices > 1 and self["--fabric-config"] is not None:
                        ttrt.runtime.set_fabric_config(
                            parse_fabric_config(self["--fabric-config"])
                        )
                    # Open a device of shape (x,y), where (x,y) is the mesh shape supplied by the flatbuffer
                    device = ttrt.runtime.open_mesh_device(mesh_options)

                    self.logging.info(f"evaluating binary={bin.file_path}")

                    pre_op_callback_runtime_config = CallbackRuntimeConfig(
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
                    post_op_callback_runtime_config = CallbackRuntimeConfig(
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

                    if not self["--disable-ttrt-callbacks"]:
                        callback_env = ttrt.runtime.DebugHooks.get(
                            pre_op_get_callback_fn(pre_op_callback_runtime_config),
                            post_op_get_callback_fn(post_op_callback_runtime_config),
                        )

                    if self["--save-artifacts"]:
                        self.artifacts.create_binary_artifacts_folder(bin)

                    if self["--emitc"]:
                        # .so are compiled such that they have the same name as flatbuffers, so we rename here
                        emitc_dylib_path = bin.file_path.replace(".ttnn", ".so")

                        # Open the dylib
                        emitc_dylib_handle = ttrt.runtime.test.open_so(emitc_dylib_path)
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

                        pre_op_callback_runtime_config.start_new_callback(
                            f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}"
                        )
                        post_op_callback_runtime_config.start_new_callback(
                            f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}"
                        )

                        # Implement optional pre_op_callback functionality here

                        program = bin.get_program(program_index)
                        # Skip private programs (e.g. subgraphs created by const-eval)
                        if program.is_private():
                            continue

                        golden_inputs = []

                        for i in range(program.num_inputs()):
                            golden_tensor = {}

                            if not self["--disable-golden"]:
                                golden_tensor = bin.fbb.get_debug_info_golden(
                                    f"input_{i}"
                                )

                            if len(golden_tensor) != 0:
                                golden_tensor = golden_tensor[0]
                                golden_tensor_torch = golden_tensor_to_torch(
                                    golden_tensor
                                )
                                golden_inputs.append(golden_tensor_torch)

                        program.populate_inputs(
                            self.torch_initializer.get_initializer(self["--init"]),
                            golden_inputs,
                        )
                        program.populate_outputs(
                            self.torch_initializer.get_initializer("zeros")
                        )

                        inputs = []
                        outputs = []
                        for i in program.input_tensors:
                            new_input = create_tensor(i)
                            inputs.append(new_input)

                        for i in program.output_tensors:
                            new_output = create_tensor(i)
                            outputs.append(new_output)

                        # load output golden tensors
                        if not self["--disable-golden"]:
                            golden_outputs_torch = []
                            for idx in range(0, len(program.output_tensors)):
                                golden_tensor = {}
                                golden_tensor = bin.fbb.get_debug_info_golden(
                                    f"output_{idx}"
                                )

                                if len(golden_tensor) != 0:
                                    golden_tensor = golden_tensor[0]
                                    golden_tensor_torch = golden_tensor_to_torch(
                                        golden_tensor
                                    )
                                    golden_outputs_torch.append(golden_tensor_torch)

                        event = None

                        # Parse the dirty tensor schedule
                        update_tensor_schedule = {}
                        if self["--dirty-tensor-schedule"]:
                            dirty_configs = self["--dirty-tensor-schedule"].split(",")

                            if not dirty_configs:
                                raise Exception(
                                    "Invalid --dirty-tensor-schedule format. Expected 'index:iterations,...'"
                                )
                            for config in dirty_configs:
                                if ":" not in config:
                                    raise Exception(
                                        f"Invalid dirty tensor configuration: '{config}'. Missing colon separator. Expected format 'index:iterations'"
                                    )
                                parts = config.split(":")
                                if len(parts) != 2:
                                    raise Exception(
                                        f"Invalid dirty tensor configuration: '{config}'. Too many colons. Expected format 'index:iterations'"
                                    )
                                try:
                                    input_idx = int(parts[0])
                                    iterations = int(parts[1])
                                except ValueError:
                                    raise Exception(
                                        f"Invalid dirty tensor configuration: '{config}'. Both index and iterations must be integers. Got '{parts[0]}' and '{parts[1]}'"
                                    )

                                if input_idx < 0:
                                    raise Exception(
                                        f"Invalid dirty tensor configuration: '{config}'. Tensor index must be non-negative. Got {input_idx}"
                                    )

                                if iterations < 0:
                                    raise Exception(
                                        f"Invalid dirty tensor configuration: '{config}'. Iterations must be non-negative. Got {iterations}"
                                    )

                                if iterations not in update_tensor_schedule:
                                    update_tensor_schedule[iterations] = []
                                update_tensor_schedule[iterations].append(input_idx)

                        # pre-upload inputs
                        inputs = convert_input_layouts(
                            device, inputs, bin.fbb, program_index
                        )

                        for loop in range(self["--loops"]):
                            self.logging.debug(
                                f"starting loop={loop+1}/{self['--loops']} for binary={bin.file_path}"
                            )
                            tracy_program_metadata = {
                                "loop_number": loop,
                                "program_index": program_index,
                                "disable_eth_dispatch": self["--disable-eth-dispatch"],
                                "enable_program_cache": self["--enable-program-cache"],
                                "dump_device_rate": self["--dump-device-rate"],
                            }
                            perf_env.set_program_metadata(str(tracy_program_metadata))

                            # Check if we need to dirty any input tensors in this iteration
                            if loop in update_tensor_schedule:
                                for input_idx in update_tensor_schedule[loop]:
                                    if input_idx < len(inputs):
                                        # Get the tensor to dirty
                                        tensor_to_dirty = inputs[input_idx]
                                        # Call the dirtyTensor function to increment the version counter
                                        expected_layout = ttrt.runtime.get_layout(
                                            bin.fbb, program_index, input_idx
                                        )
                                        perf_env.tracy_log_op_location(
                                            f"loc(arg_{input_idx})"
                                        )
                                        result_tensor = ttrt.runtime.to_layout(
                                            tensor_to_dirty,
                                            device,
                                            expected_layout,
                                            True,
                                        )
                                        inputs[input_idx] = result_tensor
                                        self.logging.info(
                                            f"Marked input tensor {input_idx} as dirty after {loop} iterations"
                                        )
                                    else:
                                        self.logging.warning(
                                            f"Cannot dirty input tensor {input_idx}, only {len(inputs)} inputs available"
                                        )

                            start_submit = time.perf_counter_ns()
                            runtime_outputs = ttrt.runtime.submit(
                                device,
                                bin.fbb,
                                program_index,
                                inputs,
                            )

                            ttrt.runtime.wait(runtime_outputs)
                            end_submit = time.perf_counter_ns()
                            e2e_duration_nanoseconds_submit = end_submit - start_submit

                            e2e_duration_nanoseconds_output = 0
                            for i, runtime_output_tensor in enumerate(runtime_outputs):
                                start_get_output = time.perf_counter_ns()
                                output_host = ttrt.runtime.to_host(
                                    runtime_output_tensor, untilize=True
                                )[0]
                                end_get_output = time.perf_counter_ns()
                                e2e_duration_nanoseconds_output += (
                                    end_get_output - start_get_output
                                )

                                ttrt.runtime.memcpy(
                                    outputs[i],
                                    output_host,
                                )
                                ttrt.runtime.deallocate_tensor(
                                    runtime_output_tensor, force=True
                                )

                                output_tensor_torch = None
                                if (
                                    self["--print-input-output-tensors"]
                                    or not self["--disable-golden"]
                                ):
                                    isEmptyTensor = not all(outputs[i].get_shape())
                                    data_buffer = bytearray(
                                        outputs[i].get_data_buffer()
                                    )
                                    if isEmptyTensor and len(data_buffer) == 0:
                                        # Create empty tensor.
                                        output_tensor_torch = torch.empty(
                                            outputs[i].get_shape(),
                                            dtype=ttrt_datatype_to_torch_dtype(
                                                outputs[i].get_dtype()
                                            ),
                                        )
                                    elif not isEmptyTensor and len(data_buffer) > 0:
                                        # Create regular tensor.
                                        output_tensor_torch = torch.frombuffer(
                                            data_buffer,
                                            dtype=ttrt_datatype_to_torch_dtype(
                                                outputs[i].get_dtype()
                                            ),
                                        ).reshape(outputs[i].get_shape())
                                    else:
                                        raise Exception(
                                            f"Failed: Tensor shape=({outputs[i].get_shape()}) and data buffer size={len(data_buffer)} do not match."
                                        )

                                # Compare program level golden.
                                golden_tensor_torch = None
                                pcc_fail = False
                                allclose_fail = False
                                if (not self["--disable-golden"]) and (
                                    i < len(golden_outputs_torch)
                                ):
                                    self.logging.debug(
                                        f"executing program level golden comparison for output_{i}"
                                    )
                                    golden_tensor_torch = golden_outputs_torch[i]
                                    if (
                                        golden_tensor_torch.shape
                                        != output_tensor_torch.shape
                                    ):
                                        raise Exception(
                                            f"Failed: program-level output doesn't match golden shape! golden_shape={golden_tensor_torch.shape}, output_shape={output_tensor_torch.shape}"
                                        )

                                    # PCC check.
                                    _, _, cal_pcc, _ = get_atol_rtol_pcc(
                                        golden_tensor_torch,
                                        output_tensor_torch,
                                        self.logging,
                                    )
                                    pcc_fail = (
                                        cal_pcc < post_op_callback_runtime_config.pcc
                                    )
                                    if not pcc_fail:
                                        self.logging.info(
                                            f"Program level golden for output_{idx} matched. pcc={cal_pcc}"
                                        )

                                    # Allclose check.
                                    # TODO(wenbinlyuTT):
                                    # 0. Fix NaNs and set equal_nan=False
                                    # 1. Quant ops may generate extreme outliers, we may need to
                                    #    implement our own allclose() that allows for a certain
                                    #    percentage of outliers.
                                    if bin.extension == ".ttm":
                                        allclose_fail = not torch.allclose(
                                            golden_tensor_torch,
                                            output_tensor_torch,
                                            rtol=self["--rtol-allclose"],
                                            atol=self["--atol-allclose"],
                                            equal_nan=True,
                                        )
                                        if not allclose_fail:
                                            self.logging.info(
                                                f"Program level golden for output_{idx} passed allclose check."
                                            )

                                golden_fail = pcc_fail or allclose_fail

                                # Save golden reference tensor if save flag is enabled.
                                if (
                                    self["--save-artifacts"]
                                    and golden_tensor_torch is not None
                                ):
                                    program_folder = f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}"
                                    self.artifacts.save_torch_tensor(
                                        program_folder,
                                        golden_tensor_torch,
                                        f"golden_output_{i}.pt",
                                    )

                                if self["--print-input-output-tensors"] or golden_fail:
                                    torch.set_printoptions(
                                        threshold=100, edgeitems=3, linewidth=120
                                    )
                                    for j, golden_input_tensor_torch in enumerate(
                                        program.input_tensors
                                    ):
                                        self.logging.info(
                                            f"Input {j}:\n{golden_input_tensor_torch}"
                                        )
                                    self.logging.info(
                                        f"Output {i}:\n{output_tensor_torch}"
                                    )
                                    if golden_tensor_torch is not None:
                                        self.logging.info(
                                            f"Golden:\n{golden_tensor_torch}"
                                        )

                                # Print the top k differences.
                                if golden_fail:
                                    top_k = self["--golden-diff-topk"]
                                    top_k_list = get_topk_diff(
                                        golden_tensor_torch,
                                        output_tensor_torch,
                                        top_k,
                                        relative=False,
                                    )
                                    self.logging.info(
                                        f"Top {top_k} absolute differences:"
                                    )
                                    for rank, (
                                        v_golden,
                                        v_output,
                                        v_diff,
                                        idx,
                                        is_int,
                                    ) in enumerate(top_k_list):
                                        if is_int:
                                            self.logging.info(
                                                f"{rank}: golden {v_golden:+.0f}, output {v_output:+.0f}, abs diff {v_diff:.0f}, idx {idx}"
                                            )
                                        else:
                                            self.logging.info(
                                                f"{rank}: golden {v_golden:+.6e}, output {v_output:+.6e}, abs diff {v_diff:.6e}, idx {idx}"
                                            )
                                    top_k_list = get_topk_diff(
                                        golden_tensor_torch,
                                        output_tensor_torch,
                                        top_k,
                                        relative=True,
                                    )
                                    self.logging.info(
                                        f"Top {top_k} relative differences:"
                                    )
                                    for rank, (
                                        v_golden,
                                        v_output,
                                        v_diff,
                                        idx,
                                        is_int,
                                    ) in enumerate(top_k_list):
                                        diff_percent = v_diff * 100
                                        if is_int:
                                            self.logging.info(
                                                f"{rank}: golden {v_golden:+.0f}, output {v_output:+.0f}, rel diff {diff_percent:4.1f}%, idx {idx}"
                                            )
                                        else:
                                            self.logging.info(
                                                f"{rank}: golden {v_golden:+.6e}, output {v_output:+.6e}, rel diff {diff_percent:4.1f}%, idx {idx}"
                                            )

                                if pcc_fail:
                                    raise PCCErrorException(
                                        f"Failed: program-level output golden comparison failed, actual_pcc={cal_pcc} < expected_pcc={post_op_callback_runtime_config.pcc}"
                                    )
                                if allclose_fail:
                                    raise AllCloseErrorException(
                                        f"Failed: program-level output golden comparison failed the allclose check"
                                    )

                            self.logging.info(
                                f"e2e_duration_nanoseconds_submit = {e2e_duration_nanoseconds_submit}"
                            )
                            self.logging.info(
                                f"e2e_duration_nanoseconds_output = {e2e_duration_nanoseconds_output}"
                            )
                            self.logging.info(
                                f"total_e2e_duration_nanoseconds_submit_plus_output = {e2e_duration_nanoseconds_submit + e2e_duration_nanoseconds_output}"
                            )
                            self.logging.debug(
                                f"finished loop={loop+1}/{self['--loops']} for binary={bin.file_path}"
                            )

                            bin.add_program_results(
                                program_index,
                                loop,
                                e2e_duration_nanoseconds_submit,
                                e2e_duration_nanoseconds_output,
                            )

                        if event is not None:
                            ttrt.runtime.wait(event)

                        # Compare to EmitC
                        if self["--emitc"]:
                            # Create symbol string to read from dylib
                            fwd_func_name = program.name

                            # pre-upload inputs
                            inputs = convert_input_layouts(
                                device, inputs, bin.fbb, program_index
                            )

                            for loop in range(self["--loops"]):
                                emitc_outs = ttrt.runtime.test.run_so_program(
                                    emitc_dylib_handle,
                                    fwd_func_name,
                                    inputs,
                                    device,
                                )
                                emitc_outs = [
                                    ttrt.runtime.to_host(emitc_out, untilize=True)[0]
                                    for emitc_out in emitc_outs
                                ]
                                self.logging.debug(
                                    f"got emitc outputs for program_index={program_index}, loop={loop}"
                                )

                                all_tensors_match = ttrt.runtime.test.compare_outs(
                                    outputs, emitc_outs
                                )

                                if not all_tensors_match:
                                    self.logging.error(
                                        "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                    )
                                    self.logging.error(outputs, emitc_outs)
                                    raise Exception(
                                        "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                    )
                            self.logging.info(
                                f"EmitC tensors match for {bin.file_path}"
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
                                    self.logging.error("Failed: output tensor all zero")

                        self.logging.debug(f"input tensors for program={program_index}")
                        for tensor in program.input_tensors:
                            self.logging.debug(f"{tensor}\n")

                        self.logging.debug(
                            f"output tensors for program={program_index}"
                        )
                        for tensor in program.output_tensors:
                            self.logging.debug(f"{tensor}\n")

                        # Read the perf data before deallocating buffers
                        device.read_device_profiler_results()

                        # if golden comparison is enabled, check golden results json file to see if test passed
                        if not self["--disable-golden"]:
                            if self["--save-artifacts"]:
                                post_op_callback_runtime_config.save_golden_report(
                                    f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}/golden_results.json"
                                )
                            # check operation level golden comparison result.
                            post_op_callback_runtime_config.check_pcc()

                        if self["--memory"]:
                            if self["--save-artifacts"]:
                                post_op_callback_runtime_config.save_memory_report(
                                    f"{self.artifacts.get_binary_folder_path(bin)}/run/program_{program_index}/memory_results.json"
                                )

                            if self["--check-memory-leak"]:
                                post_op_callback_runtime_config.check_memory_leak()

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
                finally:

                    if self["--emitc"]:
                        ttrt.runtime.test.close_so(emitc_dylib_handle)

                    ttrt.runtime.unregister_hooks()

                    # Only close the device it if was opened
                    if device is not None:
                        ttrt.runtime.close_mesh_device(device)
                        device = None

                    if num_mesh_devices > 1 and self["--fabric-config"] is not None:
                        ttrt.runtime.set_fabric_config(
                            ttrt.runtime.FabricConfig.DISABLED
                        )

        self.logging.debug(f"executing ttnn binaries")
        _execute(self.ttnn_binaries)
        self.logging.debug(f"finished executing ttnn binaries")

        self.logging.debug(f"executing ttmetal binaries")
        _execute(self.ttmetal_binaries)
        self.logging.debug(f"finished executing ttmetal binaries")

        self.logging.debug(f"executing cuda binaries")
        _execute(self.cuda_binaries)
        self.logging.debug(f"finished executing cuda binaries")

        self.logging.debug(f"------finished executing run API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing run API")

        if self["--save-artifacts"]:
            for bin in self.ttnn_binaries:
                self.artifacts.save_binary(bin, self.query)

            for bin in self.ttmetal_binaries:
                self.artifacts.save_binary(bin, self.query)

            for bin in self.cuda_binaries:
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

        for bin in self.cuda_binaries:
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
        def get_init_fns():
            return Run.TorchInitializer.init_fns

        def __init__(self, run):
            self.run = run

        def get_initializer(self, name):
            import inspect

            for func_name, func in inspect.getmembers(self, predicate=inspect.ismethod):
                if func_name == name:
                    return func

            raise Exception(f"could not find specified init function={name}")

        def randn(self, shape, dtype):
            import torch

            if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.int32):
                high = torch.iinfo(dtype).max + 1
                return torch.randint(0, high, shape, dtype=dtype)

            return torch.randn(shape, dtype=dtype)

        def arange(self, shape, dtype):
            import torch

            def volume(shape):
                v = 1
                for i in shape:
                    v *= i
                return v

            return torch.arange(volume(shape), dtype=dtype).reshape(shape)

        def zeros(self, shape, dtype):
            import torch

            return torch.zeros(shape, dtype=dtype)

        def ones(self, shape, dtype):
            import torch

            x = torch.randint(0, self.run["--ones-density"], shape)
            return torch.where(x == 0, 1.0, 0.0).to(dtype)
