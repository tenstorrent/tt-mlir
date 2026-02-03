# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from typing import Callable, List, Optional

from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn
from test_utils import shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([shape], [torch.float32])
        def clamp_scalar(
            in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
        ):
            print(f"Clamping with min: {min_arg}, max: {max_arg}")
            return builder.clamp_scalar(
                in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes", [[(32, 64), (32, 64), (32, 64)]], ids=shapes_list_str
)
def test_clamp_tensor(shapes: List[Shape], request, device):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def clamp_tensor(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 16, 16), (1, 1, 64)], ids=shape_str)
@pytest.mark.parametrize("repeat_dims", [[32, 1, 1], [1, 2, 2], [2, 3, 4], [1, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
def test_repeat(shape: Shape, repeat_dims: List[int], dtype, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([shape], [dtype])
        def repeat(
            in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.repeat(in0, repeat_dims, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 8, 1, 12, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
def test_repeat_interleave(
    shapes: List[Shape], repeats: int, dim: int, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.float32])
        def repeat_interleave(
            in0: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.repeat_interleave(
                in0, repeats=repeats, dim=dim, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
def test_concat(shapes: List[Shape], dim: int, request, device):
    # Create a wrapper function that captures dim
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def concat_wrapper(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 128), (128, 256)],
        [(32, 64), (64, 128)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_matmul(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [dtype, dtype])
        def matmul(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 128), (256, 128)],
        [(64, 128), (256, 128), (256,)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_linear(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def linear(
            in0: Operand,
            in1: Operand,
            bias_or_builder,
            builder_or_none=None,
            unit_attrs: Optional[List[str]] = None,
        ):
            if builder_or_none is not None:
                bias = bias_or_builder
                builder = builder_or_none
            else:
                bias = None
                builder = bias_or_builder

            return builder.linear(
                in0, in1, bias=bias, transpose_b=True, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
