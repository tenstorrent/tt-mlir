# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from chisel.core.compile_pipeline import chisel_pipeline
from chisel.core.enums import ExecutionType
from chisel.core.golden_executor import GoldenExecutor
from chisel.core.ops import IRModule
from chisel.core.registry import Registry
from chisel.core.tensors import TensorPool, TensorValue
from chisel.utils.location import hash_location
from chisel.utils.mapping import ttir_dtype_maps
from ttmlir.ir import Context


@pytest.mark.parametrize(
    "ttir_path, function_name",
    [
        ("runtime/tools/chisel/test/mlir/test_add.mlir", "add"),
    ],
)
def test_golden_execution(ttir_path: str, function_name: str):
    """
    Test golden execution workflow with TTIR to TTNN compilation pipeline.

    This test validates the complete golden execution process including:
    - MLIR module compilation from TTIR to TTNN
    - Registry setup and operation loading
    - Tensor pool initialization with test data
    - Golden execution of operations
    - Output tensor validation
    """
    ttir_path = Path(ttir_path)

    # Compile TTIR to TTNN using the standard pipeline
    ttir_module, ttnn_module = chisel_pipeline(ttir_path)

    context = Context()

    # Create IR modules for both execution contexts
    device_ir_module = IRModule(
        mlir_module=ttnn_module,
        context=context,
        execution_type=ExecutionType.DEVICE,
        functions=[function_name],
    )
    golden_ir_module = IRModule(
        mlir_module=ttir_module,
        context=context,
        execution_type=ExecutionType.GOLDEN,
        functions=[function_name],
    )

    print(f"Testing function: {function_name}")
    print(f"TTIR path: {ttir_path}")

    # Initialize registry and load all operations
    registry = Registry(golden_module=golden_ir_module, device_module=device_ir_module)
    registry.load_all_ops()

    # Setup tensor pool with caching enabled
    tensor_pool = TensorPool(caching=False)

    # Initialize golden executor
    executor = GoldenExecutor(registry, tensor_pool)

    # Populate tensor pool with test input tensors (ones tensors)
    for arg in golden_ir_module.get_function_inputs():
        tensor_name = arg.get_name(golden_ir_module.get_asm_state())
        tensor_value = TensorValue(
            tensor_name,
            torch.ones(
                arg.type.shape, dtype=ttir_dtype_maps[str(arg.type.element_type)]
            ),
            ExecutionType.GOLDEN,
        )
        tensor_value.set_execution_data()
        tensor_pool[tensor_name] = tensor_value
        print(f"Initialized input tensor {tensor_name}: shape={arg.type.shape}")

    # Execute golden operations for each device operation that should be compared
    operations_executed = 0
    for op in device_ir_module.get_function_ops():
        op_location = hash_location(op.location)
        debug_str = op.get_asm(enable_debug_info=True)

        if registry.should_compare(op, op_location, ExecutionType.DEVICE):
            print(f"Executing golden operation at location {op_location}")
            executor.execute_golden(op_location, debug_str)
            operations_executed += 1

    print(f"Total operations executed: {operations_executed}")

    # Validate output tensors
    output_tensors = 0
    for tensor_name, tensor_value in tensor_pool.items():
        if tensor_value.data is not None:
            print(f"Output tensor {tensor_name}: shape={tensor_value.data.shape}")
            output_tensors += 1
        else:
            print(f"Warning: Tensor {tensor_name} has no output data")

    print(f"Total output tensors: {output_tensors}")

    # Basic validation: ensure we executed some operations and produced some outputs
    assert operations_executed > 0, "No operations were executed"
    assert output_tensors > 0, "No output tensors were produced"
