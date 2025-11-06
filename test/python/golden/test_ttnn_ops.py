# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional

from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import compile_and_execute_ttnn
from test_utils import shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("ttnn")

def mish(
    in0: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.mish(in0, unit_attrs=unit_attrs)

def sigmoid(
    in0: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)

@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize(
    "test_fn",
    [
        mish,
        sigmoid,
    ],
)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if test_fn == mish and dtype == torch.float32:
        pytest.xfail(
            "Mish with float 32 causes PCC: https://github.com/tenstorrent/tt-metal/issues/31112"
        )

    compile_and_execute_ttnn(
        test_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )

def multiply(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "test_fn",
    [
        multiply,
    ],
)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, request, device
):
    compile_and_execute_ttnn(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request, device):
    def clamp_scalar(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        print(f"Clamping with min: {min_arg}, max: {max_arg}")
        return builder.clamp_scalar(
            in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
        )

    compile_and_execute_ttnn(
        clamp_scalar,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes", [[(32, 64), (32, 64), (32, 64)]], ids=shapes_list_str
)
def test_clamp_tensor(shapes: List[Shape], request, device):
    def clamp_tensor(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        clamp_tensor,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes", [[(10, 64, 32), (32, 128), (1,)]], ids=shapes_list_str
)
def test_linear(shapes: List[Shape], request, device):
    def linear(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.linear(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        linear,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 16, 16), (1, 1, 64)], ids=shape_str)
@pytest.mark.parametrize("dims", [[32, 1, 1], [1, 2, 2], [2, 3, 4], [1, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
def test_repeat(shape: Shape, dims: List[int], dtype, request, device):
    def repeat(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.repeat(in0, dims=dims, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        repeat,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    def repeat_interleave(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.repeat_interleave(
            in0, repeats=repeats, dim=dim, unit_attrs=unit_attrs
        )

    compile_and_execute_ttnn(
        repeat_interleave,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def concat(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    dim: int,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)


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
    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return concat(in0, in1, in2, dim, builder, unit_attrs)

    # Set the name for better test identification.
    concat_wrapper.__name__ = "concat"

    compile_and_execute_ttnn(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    def matmul(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        matmul,
        shapes,
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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

    dtypes = [dtype] * len(shapes)

    compile_and_execute_ttnn(
        linear,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
