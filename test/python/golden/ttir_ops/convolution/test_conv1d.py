# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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


# TODO: Remove this fixture once we support config tensors in dram for conv1d
# (mirrors the conv2d behaviour, see https://github.com/tenstorrent/tt-mlir/issues/6105).
@pytest.fixture(autouse=True)
def clear_program_cache_after_test(device):
    """Clear program cache after each conv1d test to free L1 memory."""
    yield
    conftest = sys.modules.get("conftest")
    if conftest and conftest._current_device is not None:
        conftest._current_device.clear_program_cache()


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, stride, padding, dilation, groups",
    [
        # input: (N, L, C); weight: (O, C/G, K); bias: (1, 1, O)
        # Basic kernel=3, same length via padding=1
        ((1, 32, 64), (64, 64, 3), (1, 1, 64), 1, 1, 1, 1),
        # No bias, valid padding (length shrinks)
        ((1, 32, 64), (128, 64, 3), None, 1, 0, 1, 1),
        # Stride=2 downsample
        ((1, 32, 16), (32, 16, 3), (1, 1, 32), 2, 0, 1, 1),
        # Dilation=2 atrous conv
        ((1, 64, 32), (32, 32, 3), (1, 1, 32), 1, 2, 2, 1),
        # Asymmetric padding [left, right]
        ((1, 32, 16), (16, 16, 3), (1, 1, 16), 1, (0, 1), 1, 1),
        # Grouped / depthwise conv
        ((1, 32, 64), (64, 1, 3), (1, 1, 64), 1, 1, 1, 64),
        # Batched
        ((4, 32, 32), (64, 32, 3), (1, 1, 64), 1, 1, 1, 1),
    ],
    ids=[
        "basic_k3_pad1",
        "valid_no_bias",
        "stride2",
        "dilation2",
        "asymmetric_pad",
        "depthwise",
        "batched",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_conv1d(
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
            def conv1d_wrapper(
                in0: Operand,
                weight: Operand,
                bias: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv1d(
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
            def conv1d_wrapper(
                in0: Operand,
                weight: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv1d(
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
