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
- ✅ 3D matmul: Works correctly with non-collapsed tensors (fixed in #6648)
- ❌ 3D transpose: Causes core dump due to hardcoded rank==2 assertions in permute rewriter
"""

import pytest
import torch
from typing import List

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str, Marks

pytestmark = pytest.mark.frontend("ttir")


def module_elementwise_add_3d_add(builder: TTIRBuilder):
    @builder.func([(3, 32, 64), (3, 32, 64)], [torch.float32, torch.float32])
    def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Element-wise addition operation."""
        return builder.add(in0, in1)


def module_elementwise_add_4d_add(builder: TTIRBuilder):
    @builder.func([(2, 3, 64, 32), (2, 3, 64, 32)], [torch.float32, torch.float32])
    def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Element-wise addition operation."""
        return builder.add(in0, in1)


def module_batch_matmul(builder: TTIRBuilder):
    @builder.func([(2, 32, 64), (2, 64, 32)], [torch.float32, torch.float32])
    def batch_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Batch matrix multiplication operation."""
        return builder.matmul(in0, in1)


def module_elementwise_multiply_3d_multiply(builder: TTIRBuilder):
    @builder.func([(3, 32, 64), (3, 32, 64)], [torch.float32, torch.float32])
    def elementwise_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Element-wise multiplication operation."""
        return builder.multiply(in0, in1)


def module_unary_exp_2d_exp(builder: TTIRBuilder):
    @builder.func([(3, 32, 64)], [torch.float32])
    def unary_exp(in0: Operand, builder: TTIRBuilder):
        """Unary exponential operation."""
        return builder.exp(in0)


def module_unary_exp_4d_exp(builder: TTIRBuilder):
    @builder.func([(1, 2, 32, 32)], [torch.float32])
    def unary_exp(in0: Operand, builder: TTIRBuilder):
        """Unary exponential operation."""
        return builder.exp(in0)


def module_transpose_inner_dims(builder: TTIRBuilder):
    @builder.func([(3, 32, 64)], [torch.float32])
    def transpose_inner_dims(in0: Operand, builder: TTIRBuilder):
        """Transpose operation on inner dimensions (last two dims)."""
        return builder.transpose(in0, 1, 2)


@pytest.mark.parametrize(
    "test_func,test_name",
    [
        # 3D element-wise operations (working with non-collapsed tensors)
        (module_elementwise_add_3d_add, "3d_add"),
        (module_elementwise_multiply_3d_multiply, "3d_multiply"),
        (module_unary_exp_2d_exp, "3d_exp"),
        # 4D element-wise operations (working with non-collapsed tensors)
        pytest.param(
            module_elementwise_add_4d_add,
            "4d_add",
            marks=pytest.mark.xfail(reason="Golden failure"),
        ),
        (module_unary_exp_4d_exp, "4d_exp"),
        # Batched matmul (fixed in #6648)
        (module_batch_matmul, "matmul"),
        # Operations with known issues (marked as skip)
        pytest.param(
            module_transpose_inner_dims,
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
        target=target,
        custom_pipeline=pipeline,
        test_base=f"{request.node.name}_{test_name}_{'collapsed' if collapse_tensors else 'non_collapsed'}",
        device=device,
    )
