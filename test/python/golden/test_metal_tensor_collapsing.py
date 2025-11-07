# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the --collapse-tensors-to-2d pipeline option.

This test suite validates the behavior of tensor operations when the
--collapse-tensors-to-2d option is false, allowing tensors to maintain
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
from builder.base.builder_utils import compile_and_execute_ttir
from test_utils import shape_str, Marks
from op_wrappers.eltwise import *

pytestmark = pytest.mark.frontend("ttir")


def batch_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
    """Batch matrix multiplication operation."""
    return builder.matmul(in0, in1)


def transpose_inner_dims(in0: Operand, builder: TTIRBuilder):
    """Transpose operation on inner dimensions (last two dims)."""
    return builder.transpose(in0, 1, 2)


@pytest.mark.parametrize(
    "shapes,test_func,test_name",
    [
        # 3D element-wise operations (working with non-collapsed tensors)
        ([(3, 32, 64), (3, 32, 64)], add, "3d_add"),
        ([(3, 32, 64), (3, 32, 64)], multiply, "3d_multiply"),
        ([(3, 32, 64)], exp, "3d_exp"),
        # 4D element-wise operations (working with non-collapsed tensors)
        pytest.param(
            [(2, 3, 64, 32), (2, 3, 64, 32)],
            add,
            "4d_add",
            marks=pytest.mark.xfail(reason="Golden failure"),
        ),
        ([(1, 2, 32, 32)], exp, "4d_exp"),
        # Operations with known issues (marked as skip)
        pytest.param(
            [(2, 32, 64), (2, 64, 32)],
            batch_matmul,
            "matmul",
            marks=pytest.mark.skip(
                reason="Hardcoded rank==2 assertions in matmul rewriter cause core dump"
            ),
        ),
        pytest.param(
            [(3, 32, 64)],
            transpose_inner_dims,
            "transpose",
            marks=pytest.mark.skip(
                reason="Hardcoded rank==2 assertions in permute rewriter cause core dump"
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
    device,
):
    """Test tensor operations with and without tensor collapsing to 2D."""

    pipeline_options = f"{{collapse-tensors-2d={str(collapse_tensors).lower()}}}"
    pipeline = f"ttir-to-ttmetal-pipeline{pipeline_options}"

    compile_and_execute_ttir(
        test_func,
        shapes,
        target=target,
        custom_pipeline=pipeline,
        test_base=f"{request.node.name}_{test_name}_{'collapsed' if collapse_tensors else 'non_collapsed'}",
        print_ir="ir_dump",
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )
