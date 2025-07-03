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
from chisel.core.tensors import TensorPool, TensorValue
from chisel.core.golden_executor import GoldenExecutor
from chisel.utils.location import hash_location

from chisel.utils.mapping import ttir_dtype_maps

BASE = Path("/proj_sw/user_dev/ndrakulic/chisel")
TEST_CONFIGS = {
    "dare": ["forward"],
    # "ffe_llama_l1": ["forward", "backward"],
    "ffe_mnist1": ["forward", "backward"],
    # "ffe_mnist3": ["forward", "backward"],
    # # "xla_bert": ["main"],
    # "xla_bert_l1": ["main"],
    # "xla_mnist": ["main"],
}

# Build parameter list: (golden_path, device_path, main_fn)
params = []
for model, main_fns in TEST_CONFIGS.items():
    for fn in main_fns:
        golden = BASE / model / "ttir.mlir"
        device = BASE / model / "ttnn.mlir"
        params.append((golden, device, fn))

@pytest.mark.parametrize("golden_path, device_path, main_fn", params)
def test_ir_module(golden_path: Path, device_path: Path, main_fn: str):
    print(f"Device path: {device_path}")
    print(f"Golden path: {golden_path}")
    print(f"Main function: {main_fn}")

    device_ir_module: IRModule = IRModule(
        mlir_text=device_path.read_text(),
        context=Context(),
        execution_type=ExecutionType.DEVICE,
        main_op_name=main_fn,
    )
    golden_ir_module = IRModule(
        mlir_text=golden_path.read_text(),
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
    for arg in golden_ir_module.get_function_inputs():
        tensor_value = TensorValue(
            arg.get_name(),
            torch.ones(arg.type.shape, dtype=ttir_dtype_maps[str(arg.type.element_type)]),
            ExecutionType.GOLDEN,
        )
        tensor_pool[arg.get_name()] = tensor_value

    for op in device_ir_module.main_body_ops():
        executor.execute_golden(hash_location(op.location), op.get_asm(enable_debug_info=True))

    for tensor_name, tensor_value in tensor_pool.tensors.items():
        print(f"Tensor {tensor_name}: {tensor_value.execution_data}")
