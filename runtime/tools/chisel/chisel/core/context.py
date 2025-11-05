# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from chisel.utils.writer import ReportWriter
import argparse
import pathlib
import numpy as np
import torch
import time
import os
import sys
import logging
from typing import Tuple, Literal, Callable

from chisel.utils.runtime_utils import ttir_dtype_maps
from ttmlir.ir import Context, Module, Operation

from chisel.core.ops import IRModule, get_op_inputs, get_op_outputs
from chisel.core.registry import OpGroup, Registry
from chisel.core.golden_executor import GoldenExecutor
from chisel.core.tensors import (
    TensorPool,
    TensorValue,
)
from chisel.core.enums import ExecutionType
from chisel.utils.metrics import compute_pcc, compute_abs_err, compute_rel_err
from chisel.utils.location import parse_op_location
from chisel.utils.runtime_utils import get_torch_tensor
from chisel.utils.debug import debug_wrap

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_input_refs,
    get_op_output_ref,
    get_op_debug_str,
    get_op_loc_info,
    create_owned_host_tensor,
    retrieve_tensor_from_pool,
    DataType,
    Tensor as RtTensor,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chisel")

DEBUG = True


class ChiselContext:
    """
    Main context class for Chisel that manages the execution and comparison of MLIR operations
    between golden (reference) and device implementations.

    This class handles:
    - Loading and managing MLIR modules for both golden and device execution
    - Tensor management and tracking across execution contexts
    - Operation execution and result comparison
    - Debugging and reporting utilities
    - Integration with the TTRT (Tenstorrent Runtime) backend

    Args:
        ttir_module (Module): MLIR module for the golden implementation
        ttnn_module (Module): MLIR module for the device implementation
        output_dir (pathlib.Path): Directory to store output artifacts and reports
        report_path (pathlib.Path): Path to store the execution report
        main_fn (str): Name of the main function to execute
        program_index (int, optional): Index of the program to run. Defaults to 0.
        flatbuffer_path (pathlib.Path | None, optional): Path to flatbuffer file. Defaults to None.
        function_argument_bridge_type (Literal["host", "device"], optional):
            Choose between host and device for a source of input arguments. Defaults to "host".
        caching (bool, optional): Enable tensor caching. Defaults to True.
        should_skip_op (Callable[[Operation], bool], optional):
            Function to determine if an operation should be skipped. Defaults to lambda op: False.
    """

    def __init__(
        self,
        ttir_module: Module,
        ttnn_module: Module,
        output_dir: pathlib.Path,
        report_path: pathlib.Path,
        main_fn: str,
        program_index: int = 0,
        flatbuffer_path: pathlib.Path | None = None,
        function_argument_bridge_type: Literal["host", "device"] = "host",
        caching: bool = True,
        should_skip_op: Callable[[Operation], bool] = lambda op: False,
    ):
        """
        Initialize the Chisel context with MLIR modules and runtime configuration.

        This sets up the execution environment including:
        - MLIR context and module wrapping
        - Tensor management pools
        - Execution tracking and reporting
        - Runtime integration
        """
        self.output_dir = output_dir
        self.main_fn = main_fn
        self.program_index = program_index
        self.function_argument_bridge_type = function_argument_bridge_type

        # Initialize MLIR context and load all available dialects
        self.context = Context()
        self.context.load_all_available_dialects()

        # Load and parse both golden and device MLIR modules
        logger.debug("Loading IRs...")
        self.device_ir_module = IRModule(
            mlir_module=ttnn_module,
            context=self.context,
            execution_type=ExecutionType.DEVICE,
            functions=[self.main_fn],
            current_function_name=self.main_fn,
        )
        self.golden_ir_module = IRModule(
            mlir_module=ttir_module,
            context=self.context,
            execution_type=ExecutionType.GOLDEN,
            functions=[self.main_fn],
            current_function_name=self.main_fn,
        )

        # Initialize registry and tensor pools for both execution types
        self.modules = {
            ExecutionType.DEVICE: self.device_ir_module,
            ExecutionType.GOLDEN: self.golden_ir_module,
        }

        # Set up registry, tensor pools, and executors
        self.registry = Registry(
            golden_module=self.modules[ExecutionType.GOLDEN],
            device_module=self.modules[ExecutionType.DEVICE],
            should_skip_op=should_skip_op,
        )
        self.golden_tensor_pool = TensorPool(
            caching=caching, output_dir=self.output_dir / "golden"
        )
        self.device_tensor_pool = TensorPool(
            caching=caching, output_dir=self.output_dir / "device"
        )
        self.executor = GoldenExecutor(self.registry, self.golden_tensor_pool)

        # Set up reporting
        self.report = ReportWriter(
            report_path,
            {
                ExecutionType.GOLDEN: self.golden_ir_module.get_asm_state(),
                ExecutionType.DEVICE: self.device_ir_module.get_asm_state(),
            },
        )

        self.rt_logger = RtLogger()
        self.rt_artifacts = RtArtifacts(
            logger=self.rt_logger, artifacts_folder_path=str(self.output_dir)
        )

        # Initialize operation tracking and function arguments
        self.current_device_op = None  # Tracks the currently executing device operation
        self.arg_names = [
            arg.get_name() for arg in self.device_ir_module.get_function_inputs()
        ]

        # Set up TTRT runtime if flatbuffer path is provided
        self.rt_api = None
        if flatbuffer_path is not None:
            self.flatbuffer_path = flatbuffer_path
            self.setup_ttrt()

    def setup_ttrt(self):
        """
        Initialize the TTRT environment.

        This sets up the runtime API with the provided flatbuffer binary and configuration.
        The runtime is initialized with ones initialization for tensor values.
        """
        logger.debug("Setting up TTRT...")
        # Configure runtime arguments
        args = {
            "binary": str(self.flatbuffer_path),
            "save-artifacts": True,
            "--program-index": self.program_index,
            "--init": "ones",
            "--disable-ttrt-callbacks": True,
        }

        # Initialize the runtime API with the specified configuration
        RtApi.initialize_apis()
        self.rt_api = RtApi.Run(
            args=args, logger=self.rt_logger, artifacts=self.rt_artifacts
        )

    def compare_outputs(self, op_location: Tuple[int, int]):
        """
        Compare the outputs of golden and device executions for a given operation location.

        This method:
        1. Retrieves outputs from both golden and device executions
        2. Computes comparison metrics (PCC, absolute error, relative error)
        3. Records the results in the report

        Args:
            op_location (Tuple[int, int]): The location of the operation to compare
        """
        # Get output tensors from both execution types
        device_output = self.registry.get_group_output(
            op_location, ExecutionType.DEVICE
        )
        golden_output = self.registry.get_group_output(
            op_location, ExecutionType.GOLDEN
        )

        # Extract tensor data if outputs exist
        if device_output is not None:
            device_output_name = device_output.get_name()
            device_output = self.device_tensor_pool[device_output_name].data
        else:
            device_output = None

        if golden_output is not None:
            golden_output_name = golden_output.get_name()
            golden_output = self.golden_tensor_pool[golden_output_name].data
        else:
            golden_output = None

        # Compute comparison metrics if both outputs exist
        if golden_output is not None and device_output is not None:
            pcc = compute_pcc(device_output, golden_output)
            abs_error = compute_abs_err(device_output, golden_output)
            rel_error = compute_rel_err(device_output, golden_output)
        else:
            pcc = None
            abs_error = None
            rel_error = None

        # Record comparison results in the report
        self.report.write_row(
            location=op_location,
            golden_ops=self.registry.get_group(op_location, ExecutionType.GOLDEN),
            device_ops=self.registry.get_group(op_location, ExecutionType.DEVICE),
            golden_output=self.registry.get_group_output(
                op_location, ExecutionType.GOLDEN
            ),
            device_output=self.registry.get_group_output(
                op_location, ExecutionType.DEVICE
            ),
            golden_inputs=self.registry.get_group_inputs(
                op_location, ExecutionType.GOLDEN
            ),
            device_inputs=self.registry.get_group_inputs(
                op_location, ExecutionType.DEVICE
            ),
            pcc=pcc,
            abs_error=abs_error,
            rel_error=rel_error,
            device_output_tensor=device_output,
            golden_output_tensor=golden_output,
        )

    @debug_wrap(debug=DEBUG)
    def preop(self, binary, programContext, opContext):
        """
        Pre-operation callback executed before each device operation.

        This method:
        1. Extracts operation information and location
        2. Sets up input tensors for the operation
        3. Updates tensor references in the device tensor pool

        Args:
            binary: The binary containing the operation
            programContext: Context of the current program
            opContext: Context of the current operation
        """
        # Extract debug information and operation location
        debug_str = get_op_debug_str(opContext)
        op_location = parse_op_location(get_op_loc_info(opContext))

        # Get input tensor references for the operation
        input_refs = get_op_input_refs(opContext, programContext)
        if len(input_refs) == 0:
            return

        # Find the corresponding device operation in the registry
        self.current_device_op = self.registry.find_op(
            op_location, debug_str, ExecutionType.DEVICE
        )
        if self.current_device_op is None:
            return

        # Process each input tensor
        op_inputs = get_op_inputs(self.current_device_op)
        for mlir_tensor, tensor_ref in zip(op_inputs, input_refs):
            input_name = mlir_tensor.get_name()
            if input_name in self.device_tensor_pool:
                # Update existing tensor reference if needed
                device_tensor = self.device_tensor_pool[input_name]
                if device_tensor.tensor_ref is not None:
                    continue  # Skip if tensor reference already set
                device_tensor.tensor_ref = tensor_ref
                self.function_argument_bridge(programContext, input_name)
            else:
                tensor_data = retrieve_tensor_from_pool(programContext, tensor_ref)
                data = get_torch_tensor(tensor_data)
                # Create new tensor value if it doesn't exist
                self.device_tensor_pool[input_name] = TensorValue(
                    input_name, data, ExecutionType.DEVICE, tensor_ref=tensor_ref
                )

    @debug_wrap(debug=DEBUG)
    def postop(self, binary, programContext, opContext):
        """
        Post-operation callback executed after each device operation.

        This method:
        1. Captures the operation's output tensor
        2. Stores it in the device tensor pool
        3. Triggers golden execution and comparison if needed

        Args:
            binary: The binary containing the operation
            programContext: Context of the current program
            opContext: Context of the current operation
        """
        # Extract debug information and operation location
        debug_str = get_op_debug_str(opContext)
        op_location = parse_op_location(get_op_loc_info(opContext))

        if self.current_device_op is None:
            return

        # Skip if the operation has no outputs
        if len(get_op_outputs(self.current_device_op)) == 0:
            return

        # Get the output tensor reference and data
        output_name = get_op_outputs(self.current_device_op)[0].get_name()
        output_ref = get_op_output_ref(opContext, programContext)
        if output_ref is None:
            return

        # Retrieve and store the output tensor
        output_tensor = retrieve_tensor_from_pool(programContext, output_ref)
        tensor_value = TensorValue(
            output_name,
            get_torch_tensor(output_tensor),
            ExecutionType.DEVICE,
            tensor_ref=output_ref,
        )
        self.device_tensor_pool[output_name] = tensor_value

        # Execute golden model and compare results if needed
        if self.registry.should_compare(
            self.current_device_op, op_location, ExecutionType.DEVICE
        ):
            self.executor.execute_golden(op_location, debug_str)
            if self.registry.op_groups[op_location].skip_group:
                self.skip_group(op_location, programContext)
            self.compare_outputs(op_location)

    def get_corresponding_tensors(
        self, tensor_name: str, from_type: ExecutionType, to_type: ExecutionType
    ):
        """
        Get corresponding tensors between different execution types.

        Args:
            tensor_name: Name of the tensor to find
            from_type: Source execution type (GOLDEN or DEVICE)
            to_type: Target execution type (GOLDEN or DEVICE)

        Returns:
            Tuple of (source_tensor, target_tensor)
        """
        # Get tensor location in the source execution type
        tensor_loc = self.registry.tensor_to_location[from_type][tensor_name]

        # Get MLIR tensors for both execution types
        source_mlir = self.registry.tensors[tensor_loc].get(from_type)
        target_mlir = self.registry.tensors[tensor_loc].get(to_type)

        if source_mlir is None or target_mlir is None:
            raise ValueError(f"Could not find corresponding tensors for {tensor_name}")

        # Get actual tensor values from pools
        source_pool = (
            self.golden_tensor_pool
            if from_type == ExecutionType.GOLDEN
            else self.device_tensor_pool
        )
        target_pool = (
            self.golden_tensor_pool
            if to_type == ExecutionType.GOLDEN
            else self.device_tensor_pool
        )

        source_name = source_mlir.get_name(self.modules[from_type].get_asm_state())
        target_name = target_mlir.get_name(self.modules[to_type].get_asm_state())

        source_tensor = source_pool.get(source_name)
        target_tensor = target_pool.get(target_name)

        if source_tensor is None or target_tensor is None:
            raise ValueError(
                f"Could not find tensors in pools: {source_name} or {target_name}"
            )

        return source_tensor, target_tensor

    def skip_group(self, op_location, programContext):
        """
        Skip execution of a group of operations.

        This method is called when a group of operations is marked to be skipped.
        It ensures proper tensor synchronization between device and golden models.

        Args:
            op_location: Location of the operation group
            programContext: Context of the current program
        """
        group: OpGroup = self.registry.op_groups[op_location]

        # Process inputs for the group
        device_inputs = self.registry.get_group_inputs(
            op_location, ExecutionType.DEVICE
        )
        golden_inputs = []

        # Synchronize input tensors between device and golden models
        for input in device_inputs:
            # Get corresponding golden tensor
            device_name = input.get_name(self.device_ir_module.get_asm_state())
            device_tensor, golden_tensor = self.get_corresponding_tensors(
                device_name, ExecutionType.DEVICE, ExecutionType.GOLDEN
            )

            golden_inputs.append(golden_tensor)
            golden_tensor.set_execution_data(device_tensor.data)

        # Execute golden operations with skipping
        for op in self.registry.get_group(op_location, ExecutionType.GOLDEN):
            self.executor.execute(op, skip_op=True)

        # Reset execution data for golden inputs
        for input in golden_inputs:
            input.set_execution_data()

        output = self.registry.get_group_output(op_location, ExecutionType.GOLDEN)
        output_name = output.get_name(self.golden_ir_module.get_asm_state())
        golden_tensor, device_tensor = self.get_corresponding_tensors(
            output_name, ExecutionType.GOLDEN, ExecutionType.DEVICE
        )

        # Update device tensor with golden tensor data
        device_tensor.set_execution_data(golden_tensor.execution_data)
        device_tensor.update_tensor_in_pool(programContext)

        # Reset execution data for golden output
        golden_tensor.set_execution_data()

    def function_argument_bridge(self, programContext, input_name):
        """
        Bridge function arguments between host and device execution contexts.

        This method handles the synchronization of tensor data between:
        - Host and device memory spaces
        - Golden and device execution contexts

        Args:
            programContext: Context of the current program
            input_name: Name of the input tensor to bridge
        """
        if self.function_argument_bridge_type == "host":
            # Update tensor in device memory from host data
            device_tensor = self.device_tensor_pool[input_name]
            if device_tensor.execution_data is None:
                return
            device_tensor.update_tensor_in_pool(programContext)

        elif self.function_argument_bridge_type == "device":
            # For device tensors that are function arguments, sync to golden tensors
            if input_name in self.arg_names:
                device_tensor = self.device_tensor_pool[input_name]
                # Retrieve tensor data from device
                device_data = retrieve_tensor_from_pool(
                    programContext, device_tensor.tensor_ref
                )
                torch_tensor = get_torch_tensor(device_data)

                # Create and initialize corresponding golden tensor
                golden_tensor = TensorValue(
                    input_name, torch_tensor, ExecutionType.GOLDEN
                )
                golden_tensor.set_execution_data()
                self.golden_tensor_pool[input_name] = golden_tensor

    def run(self):
        """
        Execute the runtime with the configured settings.

        This starts the execution of the loaded program using the TTRT runtime.

        Returns:
            Tuple containing result code and execution results
        """
        logger.info("Running runtime...")
        result_code, results = self.rt_api()
        return result_code, results

    def bind_callbacks(self):
        """
        Set up debug hooks for operation execution.

        This configures the pre-operation and post-operation callbacks
        that will be triggered during program execution.
        """
        callback_env_pre = DebugHooks.get(self.preop, self.postop)

    def load_inputs_from_disk(self, positional_inputs_paths):
        """
        Load input tensors from disk and initialize both golden and device tensor pools.

        Args:
            positional_inputs_paths: List of file paths containing input tensors
        """
        # Iterate through each input path and corresponding function argument
        for path, arg in zip(
            positional_inputs_paths,
            self.device_ir_module.get_function_inputs(),
            strict=True,
        ):
            tensor = torch.load(path)
            arg_name = arg.get_name()

            golden_tensor = TensorValue(arg_name, tensor, ExecutionType.GOLDEN)
            golden_tensor.set_execution_data()
            self.golden_tensor_pool[arg_name] = golden_tensor

            device_tensor = TensorValue(arg_name, tensor, ExecutionType.DEVICE)
            device_tensor.set_execution_data()
            self.device_tensor_pool[arg_name] = device_tensor
        print(self.device_tensor_pool.keys())

    def generate_random_inputs(self):
        """
        Generate random inputs for the program.
        """
        embedding_found = False
        for op in self.device_ir_module.get_function_ops():
            if op.name == "ttnn.embedding":
                embedding_found = True
                operands = op.operands
                embedding_weight = operands[1]
                shape = embedding_weight.type.shape
                if len(shape) == 2:
                    embedding_size = shape[0]
                    break

        for arg in self.device_ir_module.get_function_inputs():
            arg_name = arg.get_name()
            shape = arg.type.shape
            dtype = arg.type.element_type
            torch_dtype = ttir_dtype_maps[str(dtype)]
            if torch_dtype.is_floating_point:
                tensor = torch.randn(shape, dtype=torch_dtype)
            elif embedding_found and embedding_size is not None:
                tensor = torch.randint(0, embedding_size, shape)
                embedding_found = False
            else:
                tensor = torch.randint(
                    torch.iinfo(torch_dtype).min, torch.iinfo(torch_dtype).max, shape
                )
            self.device_tensor_pool[arg_name] = TensorValue(
                arg_name, tensor, ExecutionType.DEVICE
            )
            self.device_tensor_pool[arg_name].set_execution_data()

            self.golden_tensor_pool[arg_name] = TensorValue(
                arg_name, tensor, ExecutionType.GOLDEN
            )
            self.golden_tensor_pool[arg_name].set_execution_data()
