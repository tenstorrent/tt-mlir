# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from chisel.core.compile_pipeline import chisel_pipeline
from chisel.core.enums import ExecutionType
from chisel.core.ops import IRModule
from chisel.core.registry import Registry
from chisel.utils.writer import ReportWriter
from ttmlir.ir import Context


@pytest.mark.parametrize(
    "ttir_path, function_name",
    [
        ("runtime/tools/chisel/test/mlir/test_fusion.mlir", "transpose_matmul"),
    ],
)
def test_ir_module(ttir_path: str, function_name: str):
    ttir_path = Path(ttir_path)

    ttir_module, ttnn_module = chisel_pipeline(ttir_path)

    context = Context()

    device_ir_module = IRModule(
        mlir_module=ttnn_module,
        context=context,
        execution_type=ExecutionType.DEVICE,
        main_function_name=function_name,
    )
    golden_ir_module = IRModule(
        mlir_module=ttir_module,
        context=context,
        execution_type=ExecutionType.GOLDEN,
        main_function_name=function_name,
    )
    print(ttnn_module.operation.get_asm(enable_debug_info=True))
    print(str(ttir_module))
    report = ReportWriter(
        "report.csv",
        asm_state={
            ExecutionType.GOLDEN: golden_ir_module.get_asm_state(),
            ExecutionType.DEVICE: device_ir_module.get_asm_state(),
        },
    )

    registry = Registry(golden_module=golden_ir_module, device_module=device_ir_module)
    registry.load_all_ops()
    print(sorted(registry.op_groups.keys()))
    for loc_hash in sorted(registry.op_groups.keys()):
        print(loc_hash)
        report.write_row(
            location=loc_hash,
            golden_ops=registry.get_group(loc_hash, ExecutionType.GOLDEN),
            device_ops=registry.get_group(loc_hash, ExecutionType.DEVICE),
            golden_output=registry.get_group_output(loc_hash, ExecutionType.GOLDEN),
            device_output=registry.get_group_output(loc_hash, ExecutionType.DEVICE),
            golden_inputs=registry.get_group_inputs(loc_hash, ExecutionType.GOLDEN),
            device_inputs=registry.get_group_inputs(loc_hash, ExecutionType.DEVICE),
        )
