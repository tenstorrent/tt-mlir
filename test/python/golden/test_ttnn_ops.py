# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional

from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import compile_and_execute_ttnn
from test_utils import Marks, shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request, device):
    def clamp_scalar(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
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
    def linear_wrapper(
        *inputs_and_builder,
        unit_attrs: Optional[List[str]] = None,
    ):
        *inputs, builder = inputs_and_builder
        return builder.linear(*inputs, transpose_b=True, unit_attrs=unit_attrs)

    linear_wrapper.__name__ = "linear"
    dtypes = [dtype] * len(shapes)

    compile_and_execute_ttnn(
        linear_wrapper,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


def sum(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sum(in0, unit_attrs=unit_attrs)


def mean(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mean(in0, unit_attrs=unit_attrs)


def max(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.max(in0, unit_attrs=unit_attrs)


def min(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.min(in0, unit_attrs=unit_attrs)


reduction_ops = [
    sum,
    mean,
    max
    | Marks(
        pytest.mark.skip_config(["ttnn"]),
    ),
    min
    | Marks(
        pytest.mark.skip_config(["ttnn"]),
    ),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", reduction_ops)
def test_reduction_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shapes", [[(128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dim", [1])
def test_argmax(shapes, dim, request, device):
    def argmax(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.argmax(in0, dim, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        argmax,
        inputs_shapes=shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # [input_shape, output_shape]
        [(128, 128), (16384,)],  # Flatten 2D to 1D
        [(24,), (2, 3, 4)],  # Unflatten 1D to 3D
        [(2, 3, 4), (6, 4)],  # 3D to 2D reshape
        [(128, 128), (64, 256)],  # 2D to 2D different arrangement
        [(1, 1, 1), (1,)],  # Edge case: all dimensions are 1
        [(10,), (10,)],  # Identity reshape
        [(64, 512), (64, 1, 512)],  # Common ML pattern: expand dims
        [(256, 256), (512, 128)],  # Power of 2 reshape
        [(32, 3, 224, 224), (32, 150528)],  # Large ML pattern: batch flatten
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32, torch.uint8], ids=["f32", "i32", "ui8"]
)
def test_reshape(shapes, dtype: torch.dtype, request, device):
    input_shape, output_shape = shapes

    def reshape_wrapper(in0: Operand, builder: TTNNBuilder):
        return builder.reshape(in0, output_shape)

    compile_and_execute_ttnn(
        reshape_wrapper,
        [input_shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 1, 5, 5)], ids=shape_str)
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
def test_pad(shape: Shape, padding: List[int], value: int, request, device):
    def pad(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.pad(in0, padding=padding, value=value, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        pad,
        inputs_shapes=[shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(2, 3, 4)]], ids=shapes_list_str)
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
def test_permute(shapes: List[Shape], permutation: List[int], request, device):
    # Create a wrapper function that captures permutation
    def permute(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.permute(
            in0,
            permutation=permutation,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttnn(
        permute,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(2, 3, 4)]], ids=shapes_list_str)
def test_transpose(shapes: List[Shape], request, device):
    def transpose(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.transpose(
            in0,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttnn(
        transpose,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        ((64, 64), [0, 0], [32, 32], None),
        ((64, 64), [10, 20], [50, 60], [1, 1]),
        ((64, 64, 64), [10, 20, 30], [50, 60, 64], [2, 2, 1]),
    ],
    ids=["basic_slice", "explicit_step", "3d_slice"],
)
def test_slice(
    shape: Shape,
    begins: List[int],
    ends: List[int],
    step: List[int],
    request,
    device,
):
    def slice(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.slice(in0, begins, ends, step, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        slice,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
