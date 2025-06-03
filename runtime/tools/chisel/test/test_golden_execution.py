# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
import torch
from chisel.core.ops import IRModule
from ttmlir.ir import Context

from chisel.core.enums import ExecutionType
from chisel.core.registry import Registry
from chisel.core.tensors import TensorPool, get_function_inputs, TensorValue
from chisel.core.golden_executor import GoldenExecutor

from chisel.utils.mapping import ttir_dtype_maps


@pytest.mark.parametrize(
    "golden_path, device_path, main_fn",
    [
        # (Path("/localdev/ndrakulic/chisel/mnist/ttir.mlir"), Path("/localdev/ndrakulic/chisel/mnist/ttnn.mlir"), "forward"),
        # (Path("/localdev/ndrakulic/chisel/simple/ttir.mlir"), Path("/localdev/ndrakulic/chisel/simple/ttnn.mlir"), "main"),
        # (Path("/localdev/ndrakulic/chisel/llama_debug/ttir.mlir"), Path("/localdev/ndrakulic/chisel/llama_debug/ttnn.mlir"), "forward"),
        # (Path("/localdev/ndrakulic/chisel/llama_debug/ttir.mlir"), Path("/localdev/ndrakulic/chisel/llama_debug/ttnn.mlir"), "backward"),
        (
            Path("/localdev/ndrakulic/chisel/albert/ttir.mlir"),
            Path("/localdev/ndrakulic/chisel/albert/ttnn.mlir"),
            "main",
        ),
    ],
)
def test_ir_module(golden_path: Path, device_path: Path, main_fn: str):
    print(f"Device path: {device_path}")
    print(f"Golden path: {golden_path}")
    print(f"Main function: {main_fn}")

    device_ir_module = IRModule(
        mlir_path=device_path,
        context=Context(),
        execution_type=ExecutionType.DEVICE,
        main_op_name=main_fn,
    )
    golden_ir_module = IRModule(
        mlir_path=golden_path,
        context=Context(),
        execution_type=ExecutionType.GOLDEN,
        main_op_name=main_fn,
    )

    modules = {
        ExecutionType.DEVICE: device_ir_module,
        ExecutionType.GOLDEN: golden_ir_module,
    }

    registry = Registry(modules)
    tensor_pool = TensorPool()
    executor = GoldenExecutor(registry, tensor_pool)

    # populate tensor pool with ones tensors for all input
    for arg in get_function_inputs(
        golden_ir_module.get_main_op(), ExecutionType.GOLDEN
    ):
        tensor_value = TensorValue(
            arg.name,
            torch.ones(arg.shape, dtype=ttir_dtype_maps[arg.dtype]),
            ExecutionType.GOLDEN,
        )
        tensor_pool[arg.name] = tensor_value

    for op in device_ir_module.main_body_ops():
        executor.execute_golden(op.location, op.asm)

    for tensor_name, tensor_value in tensor_pool.tensors.items():
        print(f"Tensor {tensor_name}: {tensor_value.execution_data}")
