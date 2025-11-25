# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

# ConvTranspose2d tests
@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape",
    [
        ((1, 16, 16, 16), (16, 3, 3, 3), (1, 1, 1, 3)),  # Basic transpose conv
        ((1, 16, 16, 16), (16, 3, 3, 3), None),  # No bias
    ],
    ids=["basic", "no_bias"],
)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("output_padding", [0, 1])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_conv_transpose2d(
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Optional[Shape],
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    groups: int,
    target: str,
    request,
    device,
):
    # Skip invalid combinations where output padding >= stride
    if output_padding >= stride:
        pytest.skip("Output padding must be smaller than stride")

    def conv_transpose2d_wrapper(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv_transpose2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    conv_transpose2d_wrapper.__name__ = "conv_transpose2d"

    input_shapes = [input_shape, weight_shape]
    if bias_shape:
        input_shapes.append(bias_shape)

    compile_and_execute_ttir(
        conv_transpose2d_wrapper,
        input_shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
