# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str, dialect: str = "ttnn") -> bool:
    """Check if an op exists in the MLIR file."""
    op_pattern = f"{dialect}.{op_name}"
    with open(mlir_file, "r") as f:
        for line in f:
            if op_pattern in line:
                return True
    return False


def check_matmul_transpose(
    mlir_file: str, transpose_a: bool, transpose_b: bool
) -> bool:
    """Check if matmul op has specific transpose attributes."""
    with open(mlir_file, "r") as f:
        content = f.read()
        # Look for ttnn.matmul with the expected transpose attributes
        if "ttnn.matmul" in content:
            # Check for transpose_a attribute
            has_transpose_a = f"transpose_a = {str(transpose_a).lower()}" in content
            has_transpose_b = f"transpose_b = {str(transpose_b).lower()}" in content
            return has_transpose_a and has_transpose_b
    return False


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (64, 128),
        ],  # Will need transpose on LHS: (128, 64) @ (64, 128) -> (128, 128)
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16]])
def test_permute_matmul_fusion_enabled(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    """Test that permute is fused into matmul when flag is enabled (default)."""

    def permute_matmul(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        in0_data = torch.randn(shapes[0], dtype=dtypes[0])
        in1_data = torch.randn(shapes[1], dtype=dtypes[1])

        # Permute the first input (transpose last two dims)
        permuted = builder.permute(in0, [1, 0], unit_attrs=unit_attrs)

        # Matmul: permuted @ in1 -> (128, 64) @ (64, 128) -> (128, 128)
        result = builder.matmul(permuted, in1, unit_attrs=unit_attrs)

        # Golden: transpose first, then matmul
        golden_permuted = in0_data.permute(1, 0)
        golden_output = torch.matmul(golden_permuted, in1_data)

        builder.set_goldens(
            {in0: in0_data, in1: in1_data},
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        permute_matmul,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=["enable-permute-matmul-fusion=true"],
    )

    # When fusion is enabled, permute should NOT appear (it's fused into matmul)
    assert not check_op(output, "permute"), "Permute should be fused into matmul"
    assert check_op(output, "matmul"), "Matmul should exist"


@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 128), (64, 128)],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16]])
def test_permute_matmul_fusion_disabled(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    """Test that permute is NOT fused into matmul when flag is disabled."""

    def permute_matmul(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        in0_data = torch.randn(shapes[0], dtype=dtypes[0])
        in1_data = torch.randn(shapes[1], dtype=dtypes[1])

        # Permute the first input (transpose last two dims)
        permuted = builder.permute(in0, [1, 0], unit_attrs=unit_attrs)

        # Matmul: permuted @ in1 -> (128, 64) @ (64, 128) -> (128, 128)
        result = builder.matmul(permuted, in1, unit_attrs=unit_attrs)

        # Golden: transpose first, then matmul
        golden_permuted = in0_data.permute(1, 0)
        golden_output = torch.matmul(golden_permuted, in1_data)

        builder.set_goldens(
            {in0: in0_data, in1: in1_data},
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        permute_matmul,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=["enable-permute-matmul-fusion=false"],
    )

    # When fusion is disabled, permute should still appear
    assert check_op(
        output, "permute"
    ), "Permute should NOT be fused when flag is disabled"
    assert check_op(output, "matmul"), "Matmul should exist"


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (128, 64),
        ],  # Will need transpose on RHS: (64, 128) @ (64, 128) -> (64, 64)
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16]])
def test_permute_matmul_rhs_fusion_enabled(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    """Test that permute on RHS is fused into matmul when flag is enabled."""

    def permute_matmul_rhs(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        in0_data = torch.randn(shapes[0], dtype=dtypes[0])
        in1_data = torch.randn(shapes[1], dtype=dtypes[1])

        # Permute the second input (transpose last two dims)
        permuted = builder.permute(in1, [1, 0], unit_attrs=unit_attrs)

        # Matmul: in0 @ permuted -> (64, 128) @ (64, 128) -> (64, 64)
        result = builder.matmul(in0, permuted, unit_attrs=unit_attrs)

        # Golden: transpose second, then matmul
        golden_permuted = in1_data.permute(1, 0)
        golden_output = torch.matmul(in0_data, golden_permuted)

        builder.set_goldens(
            {in0: in0_data, in1: in1_data},
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        permute_matmul_rhs,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=["enable-permute-matmul-fusion=true"],
    )

    # When fusion is enabled, permute should NOT appear (it's fused into matmul)
    assert not check_op(output, "permute"), "Permute should be fused into matmul"
    assert check_op(output, "matmul"), "Matmul should exist"
