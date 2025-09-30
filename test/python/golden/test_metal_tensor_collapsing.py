# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the --collapse-tensors-to-2d pipeline option.

This test suite validates the behavior of tensor operations when the
--collapse-tensors-to-2d option is disabled, allowing tensors to maintain
their original dimensionality instead of being collapsed to 2D.

Current state:
- ✅ 3D addition: Works correctly with non-collapsed tensors
- ✅ 3D multiplication: Works correctly with non-collapsed tensors
- ✅ 3D exponential: Works correctly with non-collapsed tensors
- ❌ 3D matmul: Causes core dump due to hardcoded rank==2 assertions in matmul rewriter
- ❌ 3D transpose: Causes core dump due to hardcoded rank==2 assertions in permute rewriter
"""

import pytest
import torch
from typing import List

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")


def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    """Element-wise addition operation."""
    return builder.add(in0, in1)


def batch_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
    """Batch matrix multiplication operation."""
    return builder.matmul(in0, in1)


def elementwise_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
    """Element-wise multiplication operation."""
    return builder.multiply(in0, in1)


def unary_exp(in0: Operand, builder: TTIRBuilder):
    """Unary exponential operation."""
    return builder.exp(in0)


def transpose_inner_dims(in0: Operand, builder: TTIRBuilder):
    """Transpose operation on inner dimensions (last two dims)."""
    return builder.transpose(in0, 1, 2)


@pytest.mark.parametrize(
    "shapes,test_func,test_name",
    [
        # 3D element-wise operations (working with non-collapsed tensors)
        ([(3, 32, 64), (3, 32, 64)], elementwise_add, "3d_add"),
        ([(3, 32, 64), (3, 32, 64)], elementwise_multiply, "3d_multiply"),
        ([(3, 32, 64)], unary_exp, "3d_exp"),
        # 4D element-wise operations (working with non-collapsed tensors)
        ([(2, 3, 32, 64), (2, 3, 32, 64)], elementwise_add, "4d_add"),
        ([(1, 2, 32, 32)], unary_exp, "4d_exp"),
        # Operations with known issues (marked as expected failures)
        pytest.param(
            [(2, 32, 64), (2, 64, 32)],
            batch_matmul,
            "matmul",
            marks=pytest.mark.xfail(
                reason="Hardcoded rank==2 assertions in matmul rewriter"
            ),
        ),
        pytest.param(
            [(3, 32, 64)],
            transpose_inner_dims,
            "transpose",
            marks=pytest.mark.xfail(
                reason="Hardcoded 2D transpose assertions in permute rewriter"
            ),
        ),
    ],
    ids=["3d_add", "3d_multiply", "3d_exp", "4d_add", "4d_exp", "matmul", "transpose"],
)
@pytest.mark.parametrize(
    "collapse_tensors", [True, False], ids=["collapsed", "non_collapsed"]
)
@pytest.mark.parametrize("target", ["ttmetal"], ids=["ttmetal"])
def test_uncollapsed_tensors(
    shapes: List[Shape],
    test_func,
    test_name: str,
    collapse_tensors: bool,
    target: str,
    request,
):
    """Test tensor operations with and without tensor collapsing to 2D."""

    # Use pipeline options properly following the DMA test pattern
    pipeline_options = f"{{collapse-tensors-2d={str(collapse_tensors).lower()}}}"
    pipeline = f"ttir-to-ttmetal-pipeline{pipeline_options}"

    compile_ttir_to_flatbuffer(
        test_func,
        shapes,
        target=target,
        custom_pipeline=pipeline,
        test_base=f"{request.node.name}_{test_name}_{'collapsed' if collapse_tensors else 'non_collapsed'}",
        print_ir="ir_dump",
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
