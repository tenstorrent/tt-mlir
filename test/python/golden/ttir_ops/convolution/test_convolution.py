# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import Callable, List, Optional
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir
from test_utils import (
    Marks,
    shapes_list_str,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        [torch.float32, torch.float32, torch.float32],
        # skip quint8 for now. Issue: https://github.com/tenstorrent/tt-metal/issues/26568
        pytest.param(
            [
                TypeInfo(torch.quint8, scale=0.1, zero_point=128),
                TypeInfo(torch.qint8, scale=0.1, zero_point=0),
                torch.float32,
                torch.int8,
            ],
            marks=pytest.mark.skip(
                reason="Issue: https://github.com/tenstorrent/tt-metal/issues/26568"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
def test_conv2d(
    shapes: List[Shape],
    input_dtypes: List[Union[torch.dtype, TypeInfo]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
    device,
):
    def conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        conv2d,
        shapes,
        input_dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes,stride,padding,dilation,groups",
    [
        # ResNet initial 7x7 conv: stride=2, padding=3
        (
            [(1, 3, 224, 224), (64, 3, 7, 7), (1, 1, 1, 64)],
            [2, 2],
            [3, 3, 3, 3],
            [1, 1],
            1,
        ),
        # ResNet 1x1 conv: stride=1, no padding
        (
            [(1, 64, 56, 56), (64, 64, 1, 1), (1, 1, 1, 64)],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
        ),
        # ResNet 3x3 conv: stride=1, padding=1
        (
            [(1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 64)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            1,
        ),
        # ResNet bottleneck 1x1 expansion: stride=1, no padding
        (
            [(1, 64, 56, 56), (256, 64, 1, 1), (1, 1, 1, 256)],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
        ),
        # ResNet stride 2 downsampling: 3x3 conv
        (
            [(1, 64, 56, 56), (128, 64, 3, 3), (1, 1, 1, 128)],
            [2, 2],
            [1, 1, 1, 1],
            [1, 1],
            1,
        ),
        # Small test case
        (
            [(1, 16, 32, 32), (32, 16, 3, 3), (1, 1, 1, 32)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            1,
        ),
    ],
    ids=[
        "resnet_initial_7x7",
        "resnet_1x1_conv",
        "resnet_3x3_conv",
        "resnet_bottleneck_expansion",
        "resnet_stride2_downsample",
        "small_3x3",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_convolution(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    dtype: torch.dtype,
    request,
    device,
):
    """Test the ttir.convolution op with various ResNet-style configurations"""

    def convolution(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.convolution(
            in0,
            weight,
            bias,
            window_strides=stride,
            padding=padding,
            weight_dilation=dilation,
            feature_group_count=groups,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        convolution,
        shapes,
        [dtype] * len(shapes),
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes,stride,padding,dilation,groups",
    [
        # Depthwise convolution (groups = input_channels)
        (
            [(1, 32, 28, 28), (32, 1, 3, 3), (1, 1, 1, 32)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            32,
        ),
        # Group convolution (4 groups)
        (
            [(1, 64, 32, 32), (64, 16, 3, 3), (1, 1, 1, 64)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            4,
        ),
        # Dilated convolution
        (
            [(1, 32, 32, 32), (64, 32, 3, 3), (1, 1, 1, 64)],
            [1, 1],
            [2, 2, 2, 2],
            [2, 2],
            1,
        ),
    ],
    ids=["depthwise_conv", "group_conv_4groups", "dilated_conv"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_convolution_groups_dilation(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    dtype: torch.dtype,
    request,
    device,
):
    """Test convolution with group and dilation patterns"""

    def convolution_grouped(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.convolution(
            in0,
            weight,
            bias,
            window_strides=stride,
            padding=padding,
            weight_dilation=dilation,
            feature_group_count=groups,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        convolution_grouped,
        shapes,
        [dtype] * len(shapes),
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (3, 8, 8, 256),
            (256, 256, 3, 3),
            (1, 1, 1, 256),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize(
    "stride,padding,output_padding,dilation,groups", [(1, 0, 0, 1, 1)]
)
def test_conv_transpose2d(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    groups: int,
    request,
    device,
):
    def conv_transpose2d(
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

    compile_and_execute_ttir(
        conv_transpose2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 64, 3, 3),
            (1, 1, 1, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("stride", [[2, 1]])
@pytest.mark.parametrize("dilation", [[2, 1]])
@pytest.mark.parametrize("padding", [[2, 1]])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_conv2d(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    target: str,
    request,
    device,
):
    """Test hoisted conv2d operation"""

    def hoisted_conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=["ttir.should_hoist"],
        )

    hoisted_conv2d.__name__ = "hoisted_conv2d"

    compile_and_execute_ttir(
        hoisted_conv2d,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
