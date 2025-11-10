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


def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


def rotate(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_torch_golden(
    input_data: torch.Tensor,
    cos_data: torch.Tensor,
    sin_data: torch.Tensor,
) -> torch.Tensor:
    # Unsqueeze cos and sin
    cos_unsqueezed = torch.unsqueeze(cos_data, dim=0)
    sin_unsqueezed = torch.unsqueeze(sin_data, dim=0)

    # Rotate input
    rotated = rotate(input_data)

    # Final computation: input * cos + rotated * sin
    return input_data * cos_unsqueezed + rotated * sin_unsqueezed


def build_ttir(
    input: Operand,
    cos_input: Operand,
    sin_input: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    unsqueezed_shape = [1] + sin_input.type.shape
    cos_unsqueezed = builder.reshape(cos_input, shape=unsqueezed_shape)

    # Multiply input with sin
    unrotated = builder.multiply(input, cos_unsqueezed, unit_attrs=unit_attrs)

    last_dim = input.type.shape[-1]
    half_dim = last_dim // 2

    # Slice second half of input
    begins = [0, 0, 0, half_dim]
    ends = input.type.shape[:3] + [last_dim]
    slice1 = builder.slice(input, begins=begins, ends=ends, step=[1, 1, 1, 1])

    # Negate the second half
    neg_slice = builder.neg(slice1, unit_attrs=unit_attrs)

    # Slice first half of input
    begins = [0, 0, 0, 0]
    ends = input.type.shape[:3] + [half_dim]
    slice2 = builder.slice(input, begins=begins, ends=ends, step=[1, 1, 1, 1])

    # Concat negated second half with first half
    input_rotated = builder.concat([neg_slice, slice2], dim=3)

    # Unsqueeze sin
    sin_unsqueezed = builder.reshape(sin_input, shape=unsqueezed_shape)

    # Multiply rotated with broadcasted cos
    rotated = builder.multiply(input_rotated, sin_unsqueezed, unit_attrs=unit_attrs)

    # Add the two products
    return builder.add(unrotated, rotated, unit_attrs=unit_attrs)


@pytest.mark.parametrize(
    "shapes",
    [
        # prefill
        [
            (1, 32, 1024, 64),  # input
            (1, 1024, 64),  # cos input
            (1, 1024, 64),  # sin input
        ],
        [
            (1, 8, 1, 64),  # input
            (1, 1, 64),  # cos input
            (1, 1, 64),  # sin input
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 3])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rotary_embedding(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test rotary position embedding (RoPE) pattern.
    This test implements the RoPE operation as a sequence of TTIR ops:
    - Reshape cos/sin to unsqueeze first dimension
    - Split input into two halves along last dimension
    - Rotate: concat(neg(second_half), first_half)
    - Multiply and add: input * cos + rotated * sin
    """

    def rotary_embedding(
        input: Operand,
        cos_input: Operand,
        sin_input: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensors
        input_data = torch.randn(shapes[0], dtype=dtypes[1])
        cos_data = torch.randn(shapes[1], dtype=dtypes[0])
        sin_data = torch.randn(shapes[2], dtype=dtypes[2])

        golden_output = build_torch_golden(input_data, cos_data, sin_data)

        result = build_ttir(input, cos_input, sin_input, builder, unit_attrs=unit_attrs)

        builder.set_goldens(
            {input: input_data, cos_input: cos_data, sin_input: sin_data},
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        rotary_embedding,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "rotary_embedding")
    if shapes[0][-2] % 32 != 0:
        assert check_op(output, "slice_static")


@pytest.mark.parametrize(
    "shapes",
    [
        # decode
        pytest.param(
            [
                (1, 8, 1, 64),  # input
                (1, 1, 64),  # cos input
                (1, 1, 64),  # sin input
            ],
            # Mark as fail becase of https://github.com/tenstorrent/tt-metal/issues/31567
            # I will follow up with decomposition to slice tensor until there is
            # some resolution from metal team.
            # Issue for decompositon https://github.com/tenstorrent/tt-mlir/issues/5621
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 3])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rotary_embedding(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test rotary position embedding (RoPE) pattern.
    This test implements the RoPE operation as a sequence of TTIR ops:
    - Reshape cos/sin to unsqueeze first dimension
    - Split input into two halves along last dimension
    - Rotate: concat(neg(second_half), first_half)
    - Multiply and add: input * cos + rotated * sin
    """

    def rotary_embedding(
        input: Operand,
        cos_input: Operand,
        sin_input: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensors
        input_data = torch.randn(shapes[0], dtype=dtypes[1])
        cos_data = torch.randn(shapes[1], dtype=dtypes[0])
        sin_data = torch.randn(shapes[2], dtype=dtypes[2])

        golden_output = build_torch_golden(input_data, cos_data, sin_data)

        result = build_ttir(input, cos_input, sin_input, builder, unit_attrs=unit_attrs)

        builder.set_goldens(
            {input: input_data, cos_input: cos_data, sin_input: sin_data},
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        rotary_embedding,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )

    assert check_op(output, "rotary_embedding")
