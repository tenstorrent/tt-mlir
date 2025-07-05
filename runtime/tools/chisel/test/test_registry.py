# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from chisel.core.ops import IRModule
from ttmlir.ir import Context
from chisel.core.enums import ExecutionType
from chisel.core.registry import Registry
from chisel.core.ops import get_op_outputs
from chisel.utils.writer import ReportWriter
from chisel.utils.location import hash_location

BASE = Path("/proj_sw/user_dev/ndrakulic/chisel")
TEST_CONFIGS = {
    # "dare": ["forward"],
    # "ffe_llama_l1": ["forward", "backward"],
    # "ffe_mnist1": ["forward", "backward"],
    "ffe_mnist3": ["forward", "backward"],
    # "xla_bert": ["main"],
    # "xla_bert_l1": ["main"],
    # "xla_mnist": ["main"],
    # "opt125m": ["main"],
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
    context = Context()

    device_ir_module = IRModule(
        mlir_text=device_path.read_text(),
        context=context,
        execution_type=ExecutionType.DEVICE,
        main_function_name=main_fn,
    )
    golden_ir_module = IRModule(
        mlir_text=golden_path.read_text(),
        context=context,
        execution_type=ExecutionType.GOLDEN,
        main_function_name=main_fn,
    )

    modules = {
        ExecutionType.DEVICE: device_ir_module,
        ExecutionType.GOLDEN: golden_ir_module,
    }

    report = ReportWriter("test.csv")

    registry = Registry(modules)

    for op in device_ir_module.get_function_ops():
        loc_hash = hash_location(op.location)
        registry.init_ops_until(loc_hash)
        if not registry.should_compare(op, loc_hash, ExecutionType.DEVICE):
            continue

        report.write_row(
            location=loc_hash,
            golden_ops=registry.get_group(loc_hash, ExecutionType.GOLDEN),
            device_ops=registry.get_group(loc_hash, ExecutionType.DEVICE),
            golden_output=registry.get_group_output(loc_hash, ExecutionType.GOLDEN),
            device_output=registry.get_group_output(loc_hash, ExecutionType.DEVICE),
            golden_inputs=registry.get_group_inputs(loc_hash, ExecutionType.GOLDEN),
            device_inputs=registry.get_group_inputs(loc_hash, ExecutionType.DEVICE),
        )
