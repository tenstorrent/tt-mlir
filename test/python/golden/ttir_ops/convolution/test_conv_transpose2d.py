# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, stride, padding, output_padding, dilation, groups",
    [
        # === GAN Generator patterns (DCGAN, StyleGAN, BigGAN) ===
        # DCGAN early upsample: 4x4→8x8, kernel=4, stride=2 (most common GAN pattern)
        ((1, 4, 4, 512), (512, 256, 4, 4), (1, 1, 1, 256), 2, 1, 0, 1, 1),
        # DCGAN mid-stage: 8x8→16x16
        ((1, 8, 8, 256), (256, 128, 4, 4), (1, 1, 1, 128), 2, 1, 0, 1, 1),
        # DCGAN final to RGB: 32x32→64x64 (no bias before tanh activation)
        ((1, 32, 32, 64), (64, 3, 4, 4), None, 2, 1, 0, 1, 1),
        # === U-Net Decoder patterns (medical imaging, segmentation) ===
        # U-Net 2x2 upsample deep layer: 16x16→32x32
        ((1, 16, 16, 512), (512, 256, 2, 2), (1, 1, 1, 256), 2, 0, 0, 1, 1),
        # U-Net 2x2 mid decoder: 32x32→64x64
        ((1, 32, 32, 256), (256, 128, 2, 2), (1, 1, 1, 128), 2, 0, 0, 1, 1),
        # U-Net shallow decoder: 64x64→128x128 (no bias before skip connection concat)
        ((1, 64, 64, 128), (128, 64, 2, 2), None, 2, 0, 0, 1, 1),
        # === FCN / Semantic Segmentation (FCN-8s, FCN-16s, DeepLab) ===
        # FCN-style 4x4 kernel with padding (feature map alignment)
        ((1, 14, 14, 512), (512, 256, 4, 4), (1, 1, 1, 256), 2, 1, 0, 1, 1),
        # FCN 2x upsample from pool5: 7x7→14x14
        ((1, 7, 7, 512), (512, 512, 4, 4), (1, 1, 1, 512), 2, 1, 0, 1, 1),
        # Segmentation final upsample to num_classes (21 VOC, 19 Cityscapes)
        ((1, 64, 64, 64), (64, 21, 2, 2), (1, 1, 1, 21), 2, 0, 0, 1, 1),
        # === SegNet / Output padding patterns (exact size recovery) ===
        # 3x3 kernel with output_padding=1 (matches encoder pooling indices)
        ((1, 16, 16, 256), (256, 128, 3, 3), (1, 1, 1, 128), 2, 1, 1, 1, 1),
        # Stride=4 with output_padding (large upsampling factor)
        ((1, 8, 8, 512), (512, 256, 5, 5), (1, 1, 1, 256), 4, 2, 1, 1, 1),
        # === Super Resolution (SRCNN variants, ESPCN alternatives) ===
        # 2x upscale: 64x64→128x128
        ((1, 64, 64, 64), (64, 64, 4, 4), None, 2, 1, 0, 1, 1),
        # 4x upscale with large kernel: 32x32→128x128
        ((1, 32, 32, 64), (64, 64, 8, 8), (1, 1, 1, 64), 4, 2, 0, 1, 1),
        # === VAE / Autoencoder Decoder ===
        # VAE bottleneck upsample: 8x8→16x16
        ((1, 8, 8, 256), (256, 128, 3, 3), (1, 1, 1, 128), 2, 1, 1, 1, 1),
        # Autoencoder reconstruction mid-layer: 16x16→32x32
        ((1, 16, 16, 128), (128, 64, 3, 3), (1, 1, 1, 64), 2, 1, 1, 1, 1),
        # === Stride=1 patterns (feature refinement without upsampling) ===
        # Refinement layer: same spatial size, channel reduction
        ((1, 32, 32, 256), (256, 128, 3, 3), (1, 1, 1, 128), 1, 1, 0, 1, 1),
        # === Batched Training scenarios ===
        # Batch=8 GAN generator training
        ((8, 8, 8, 256), (256, 128, 4, 4), (1, 1, 1, 128), 2, 1, 0, 1, 1),
        # Batch=16 segmentation decoder training (no bias before BN)
        ((16, 16, 16, 256), (256, 128, 2, 2), None, 2, 0, 0, 1, 1),
    ],
    ids=[
        "dcgan_4x4_to_8x8",
        "dcgan_8x8_to_16x16",
        "dcgan_to_rgb_no_bias",
        "unet_2x2_deep",
        "unet_2x2_mid",
        "unet_2x2_shallow_no_bias",
        "fcn_4x4_deep",
        "fcn_pool5_upsample",
        "fcn_segmentation_output",
        "segnet_3x3_output_padding",
        "stride4_output_padding",
        "superres_2x_no_bias",
        "superres_4x_large_kernel",
        "vae_bottleneck",
        "autoencoder_mid",
        "refinement_stride1",
        "batch8_gan_training",
        "batch16_segmentation_training",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
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

    else:
        input_shapes = [input_shape, weight_shape]
        input_types = [dtype, dtype]

        def module(builder: TTIRBuilder):
            @builder.func([input_shape, weight_shape], [dtype, dtype])
            def conv_transpose2d_wrapper(
                in0: Operand,
                weight: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv_transpose2d(
                    in0,
                    weight,
                    None,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
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
