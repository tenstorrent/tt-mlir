# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional, Tuple, Union
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, stride, padding, dilation, groups",
    [
        # === Core patterns (80% of production usage) ===
        # ResNet stem: 7x7, stride=2, large input, RGB
        ((1, 224, 224, 3), (64, 3, 7, 7), (1, 1, 1, 64), 2, 3, 1, 1),
        # Most common: 3x3 stride=1, no bias (before BatchNorm)
        ((1, 56, 56, 64), (128, 64, 3, 3), None, 1, 1, 1, 1),
        # Bottleneck 1x1 reduce: deep channels
        ((1, 14, 14, 1024), (256, 1024, 1, 1), (1, 1, 1, 256), 1, 0, 1, 1),
        # Bottleneck 1x1 expand: very deep channels (2048), smallest spatial
        ((1, 7, 7, 512), (2048, 512, 1, 1), (1, 1, 1, 2048), 1, 0, 1, 1),
        # Stage transition: 3x3 stride=2 downsample
        ((1, 28, 28, 256), (512, 256, 3, 3), (1, 1, 1, 512), 2, 1, 1, 1),
        # === Efficient architectures (MobileNet/EfficientNet) ===
        # Depthwise 3x3 stride=1
        ((1, 112, 112, 64), (64, 1, 3, 3), (1, 1, 1, 64), 1, 1, 1, 64),
        # Depthwise 3x3 stride=2 (spatial reduction)
        ((1, 56, 56, 128), (128, 1, 3, 3), (1, 1, 1, 128), 2, 1, 1, 128),
        # === Modern architectures (ConvNeXt/ViT) ===
        # Patchify stem: 4x4, stride=4, non-overlapping
        ((1, 224, 224, 3), (96, 3, 4, 4), (1, 1, 1, 96), 4, 0, 1, 1),
        # Large kernel depthwise: 7x7
        ((1, 14, 14, 384), (384, 1, 7, 7), (1, 1, 1, 384), 1, 3, 1, 384),
        # === Inception patterns ===
        # 5x5 kernel
        ((1, 32, 32, 64), (96, 64, 5, 5), (1, 1, 1, 96), 1, 2, 1, 1),
        # Asymmetric 1x3 (factorized)
        ((1, 32, 32, 64), (96, 64, 1, 3), (1, 1, 1, 96), 1, (0, 1), 1, 1),
        # === Semantic segmentation (DeepLab ASPP) ===
        # Atrous conv dilation=6
        ((1, 32, 32, 256), (256, 256, 3, 3), (1, 1, 1, 256), 1, 6, 6, 1),
        # Atrous conv dilation=12
        ((1, 32, 32, 256), (256, 256, 3, 3), (1, 1, 1, 256), 1, 12, 12, 1),
        # === Grouped convolutions (ResNeXt) ===
        # Cardinality=32 grouped conv
        ((1, 14, 14, 256), (256, 8, 3, 3), (1, 1, 1, 256), 1, 1, 1, 32),
        # === Object detection ===
        # YOLO stem: 640x640 input
        ((1, 640, 640, 3), (32, 3, 3, 3), (1, 1, 1, 32), 2, 1, 1, 1),
        # === Training batch sizes ===
        # Batch=16 (common training)
        ((16, 32, 32, 64), (128, 64, 3, 3), (1, 1, 1, 128), 1, 1, 1, 1),
        # Batch=32 (large training)
        ((32, 32, 32, 64), (128, 64, 3, 3), None, 1, 1, 1, 1),
    ],
    ids=[
        "resnet_stem_7x7",
        "resnet_3x3_no_bias",
        "bottleneck_1x1_reduce",
        "bottleneck_1x1_expand_2048ch",
        "downsample_3x3_s2",
        "depthwise_3x3",
        "depthwise_3x3_s2",
        "convnext_patchify_4x4_s4",
        "convnext_depthwise_7x7",
        "inception_5x5",
        "inception_1x3_asymmetric",
        "aspp_dilation6",
        "aspp_dilation12",
        "resnext_groups32",
        "yolo_640x640",
        "batch16_training",
        "batch32_training_no_bias",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_conv2d(
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Optional[Shape],
    stride: int,
    padding: Union[int, Tuple[int, int]],
    dilation: int,
    groups: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    if bias_shape:
        input_shapes = [input_shape, weight_shape, bias_shape]
        input_types = [dtype, dtype, dtype]

        def module(builder: TTIRBuilder):
            @builder.func(input_shapes, input_types)
            def conv2d_wrapper(
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

    else:
        input_shapes = [input_shape, weight_shape]
        input_types = [dtype, dtype]

        def module(builder: TTIRBuilder):
            @builder.func(input_shapes, input_types)
            def conv2d_wrapper(
                in0: Operand,
                weight: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv2d(
                    in0,
                    weight,
                    None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    unit_attrs=unit_attrs,
                )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
