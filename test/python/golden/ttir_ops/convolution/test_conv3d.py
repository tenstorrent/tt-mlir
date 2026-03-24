# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import sys
import torch
from typing import List, Optional, Tuple, Union
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


@pytest.fixture(autouse=True)
def clear_program_cache_after_test(device):
    """Clear program cache after each conv3d test to free L1 memory.

    Conv operations allocate tensors in L1 small that are only deallocated
    when the program cache is cleared. This fixture ensures the program cache
    is cleared after each test to prevent OOM errors from accumulated allocations.
    """
    yield
    conftest = sys.modules.get("conftest")
    if conftest and conftest._current_device is not None:
        conftest._current_device.clear_program_cache()


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, stride, padding, groups",
    [
        # Basic 3x3x3 kernel, no padding, no bias
        ((1, 8, 28, 28, 4), (16, 4, 3, 3, 3), None, [1, 1, 1], [0, 0, 0], 1),
        # 3x3x3 kernel with bias
        (
            (1, 8, 28, 28, 4),
            (16, 4, 3, 3, 3),
            (1, 1, 1, 1, 16),
            [1, 1, 1],
            [0, 0, 0],
            1,
        ),
        # Stride=2 with padding
        (
            (1, 8, 28, 28, 16),
            (32, 16, 3, 3, 3),
            (1, 1, 1, 1, 32),
            [2, 2, 2],
            [1, 1, 1],
            1,
        ),
        # 3x3x3 with same-padding (padding=1)
        (
            (1, 8, 28, 28, 32),
            (32, 32, 3, 3, 3),
            (1, 1, 1, 1, 32),
            [1, 1, 1],
            [1, 1, 1],
            1,
        ),
        # Larger 5x5x5 kernel
        ((1, 16, 32, 32, 8), (32, 8, 5, 5, 5), None, [1, 1, 1], [0, 0, 0], 1),
        # Stride=2, no padding, no bias (downsampling)
        ((1, 8, 28, 28, 32), (64, 32, 3, 3, 3), None, [2, 2, 2], [0, 0, 0], 1),
        # 1x1x1 kernel (pointwise 3D convolution)
        (
            (1, 8, 16, 16, 64),
            (128, 64, 1, 1, 1),
            (1, 1, 1, 1, 128),
            [1, 1, 1],
            [0, 0, 0],
            1,
        ),
        # 3x1x1 kernel, stride=[2,1,1] (temporal downsampling)
        ((1, 5, 64, 64, 192), (192, 192, 3, 1, 1), None, [2, 1, 1], [0, 0, 0], 1),
    ],
    ids=[
        "basic_3x3x3_no_bias",
        "basic_3x3x3_with_bias",
        "stride2_with_padding",
        "same_padding_3x3x3",
        "large_kernel_5x5x5",
        "stride2_downsample_no_bias",
        "pointwise_1x1x1",
        "temporal_downsampling_192ch_s211",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_conv3d(
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Optional[Shape],
    stride: List[int],
    padding: List[int],
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
            def conv3d_wrapper(
                in0: Operand,
                weight: Operand,
                bias: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv3d(
                    in0,
                    weight,
                    bias,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    unit_attrs=unit_attrs,
                )

    else:
        input_shapes = [input_shape, weight_shape]
        input_types = [dtype, dtype]

        def module(builder: TTIRBuilder):
            @builder.func(input_shapes, input_types)
            def conv3d_wrapper(
                in0: Operand,
                weight: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv3d(
                    in0,
                    weight,
                    None,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    unit_attrs=unit_attrs,
                )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
