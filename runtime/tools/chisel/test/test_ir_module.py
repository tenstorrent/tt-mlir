# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from chisel.core.ops import IRModule, attr_type_set
from ttmlir.ir import Context
from chisel.core.enums import ExecutionType


@pytest.mark.parametrize(
    "path, execution_type, main_fn",
    [
        (
            Path("/localdev/ndrakulic/chisel/llama_debug/ttnn.mlir"),
            ExecutionType.DEVICE,
            "backward",
        ),
        (
            Path("/localdev/ndrakulic/chisel/llama_debug/ttir.mlir"),
            ExecutionType.GOLDEN,
            "backward",
        ),
        (
            Path("/localdev/ndrakulic/chisel/mnist/ttir.mlir"),
            ExecutionType.GOLDEN,
            "forward",
        ),
        (
            Path("/localdev/ndrakulic/chisel/simple/ttir.mlir"),
            ExecutionType.GOLDEN,
            "main",
        ),
        (
            Path("/localdev/ndrakulic/chisel/llama_debug/ttir.mlir"),
            ExecutionType.GOLDEN,
            "forward",
        ),
        (
            Path("/localdev/ndrakulic/chisel/albert/ttir.mlir"),
            ExecutionType.GOLDEN,
            "main",
        ),
        (
            Path("/localdev/ndrakulic/chisel/mnist/ttnn.mlir"),
            ExecutionType.DEVICE,
            "forward",
        ),
        (
            Path("/localdev/ndrakulic/chisel/simple/ttnn.mlir"),
            ExecutionType.DEVICE,
            "main",
        ),
        (
            Path("/localdev/ndrakulic/chisel/llama_debug/ttnn.mlir"),
            ExecutionType.DEVICE,
            "forward",
        ),
        (
            Path("/localdev/ndrakulic/chisel/simple/ttnn.mlir"),
            ExecutionType.DEVICE,
            "main",
        ),
    ],
)
def test_ir_module(path: Path, execution_type: ExecutionType, main_fn: str):
    ir_module = IRModule(
        mlir_path=path,
        context=Context(),
        execution_type=execution_type,
        main_op_name=main_fn,
    )
