# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from chisel.core.ops import IRModule, attr_type_set
from ttmlir.ir import Context
from chisel.core.enums import ExecutionType
from chisel.core.registry import Registry

BASE = Path("/proj_sw/user_dev/ndrakulic/chisel")
TEST_CONFIGS = {
    # "dare": ["forward"],
    # "ffe_llama_l1": ["forward", "backward"],
    # "ffe_mnist1": ["forward", "backward"],
    # "ffe_mnist3": ["forward", "backward"],
    # # "xla_bert": ["main"],
    # "xla_bert_l1": ["main"],
    # "xla_mnist": ["main"],
    "opt125m": ["main"],
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

    device_ir_module = IRModule(
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
    # import pdb; pdb.set_trace()
    registry.dump_registry(out_path=device_path.with_suffix(".xlsx"))
