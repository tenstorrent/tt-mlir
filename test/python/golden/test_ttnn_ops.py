# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import compile_and_execute_ttnn
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttnn")

# Passes
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


# Passes
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


# Passes
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


def matmul(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.matmul(in0, in1, unit_attrs=unit_attrs)


def sum(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sum(in0, unit_attrs=unit_attrs)


def mean(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mean(in0, unit_attrs=unit_attrs)


def max(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.max(in0, unit_attrs=unit_attrs)


def min(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.min(in0, unit_attrs=unit_attrs)


def reshape(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    # Calculate total elements in the input tensor
    input_shape = builder.get_shape(in0)
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim

    # Reshape to a 1D tensor with all elements
    new_shape = [int(total_elements)]  # This must be a list of integers
    return builder.reshape(in0, new_shape, unit_attrs=unit_attrs)


def transpose(
    in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.transpose(in0, unit_attrs=unit_attrs)


def squeeze(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.squeeze(in0, unit_attrs=unit_attrs)


# Fails
@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dim_arg", [0])
@pytest.mark.parametrize("keep_dim", [False])
def test_prod(shape: Shape, dim_arg: int, keep_dim: bool, request, device):
    def prod(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.prod(in0, [dim_arg], keep_dim, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        prod,
        [shape],
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


# Passes
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


# Passes
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


# Passes
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


# Fails
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
        builder: TTNNBuilder,
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

    compile_and_execute_ttnn(
        conv2d,
        shapes,
        input_dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


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
@pytest.mark.parametrize("stride", [[2, 1]])
@pytest.mark.parametrize("dilation", [[2, 1]])
@pytest.mark.parametrize("padding", [[2, 1]])
@pytest.mark.parametrize("groups", [2])
def test_conv2d_consteval(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
    device,
):
    def conv2d_consteval(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTNNBuilder,
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

    compile_and_execute_ttnn(
        conv2d_consteval,
        shapes,
        argument_types_string="conv2d_consteval=input,parameter,parameter",
        test_base=request.node.name,
        device=device,
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
        builder: TTNNBuilder,
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

    compile_and_execute_ttnn(
        conv_transpose2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [([2, 2], [2, 2], [1, 1], [0, 0, 0, 0], False)],
)
@pytest.mark.parametrize("shape", [(1, 128, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_max_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    request,
    device,
):
    def max_pool2d(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.max_pool2d(
            in0,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttnn(
        max_pool2d,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode,count_include_pad",
    [
        ([2, 2], [2, 2], [1, 1], [1, 1, 1, 1], False, True),
        (
            [2, 2],
            [1, 1],
            [1, 1],
            [1, 1, 1, 1],
            True,
            False,
        ),  # This test will produce a different output if count_include_pad is True for spatial dims (31, 31)
    ],
)
@pytest.mark.parametrize("shape", [(1, 31, 31, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_avg_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    request,
    device,
):
    def avg_pool2d(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.avg_pool2d(
            in0,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttnn(
        avg_pool2d,
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
            (1, 64, 32, 32),  # input tensor: (N, C, H, W)
            (64,),  # scale (gamma)
            (64,),  # offset (beta)
            (64,),  # mean
            (64,),  # variance
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 5])
@pytest.mark.parametrize("dimension", [1])  # channel dimension
@pytest.mark.parametrize("epsilon", [1e-5])
def test_batch_norm(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    dimension: int,
    epsilon: float,
    request,
    device,
):
    def batch_norm(
        in0: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        builder,
        unit_attrs: Optional[List[str]] = None,
    ):

        return builder.batch_norm(
            in0,
            scale,
            offset,
            mean,
            variance,
            epsilon=epsilon,
            dimension=dimension,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttnn(
        batch_norm,
        shapes,
        dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Fails
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


# TODO (ctod): These three nullary tensor creation ops can probably be combined in some way.
@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int32], ids=["bf16", "f32", "i32"]
)
def test_zeros(shape: Shape, dtype: torch.dtype, request, device):
    def zeros(builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.zeros(shape, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        zeros,
        inputs_shapes=[],
        inputs_types=[],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
def test_ones(shape: Shape, request, device):
    def ones(builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.ones(shape, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        ones,
        inputs_shapes=[],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor(1, dtype=torch.uint8),
        torch.tensor([1, 2, 3, 4], dtype=torch.uint16),
        torch.tensor([1, 2, 3, 4], dtype=torch.uint32),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.bfloat16),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    ],
    ids=[
        "scalar_int-uint8",
        "1d_int-uint16",
        "1d_int-uint32",
        "2d_int-int32",
        "1d_float-bf16",
        "1d_float-f32",
        "torch_float_tensor-float32",
    ],
)
def test_constant(tensor, request, device):
    def constant(builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.constant(tensor, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        constant,
        [],
        [],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_callable_initialization_basic(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    """Basic test demonstrating callable initialization with torch.zeros and torch.ones"""

    def test_with_basic_callables(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_goldens({in0: torch.zeros, in1: torch.ones})
        result = builder.add(in0, in1, unit_attrs=unit_attrs)
        return result

    compile_and_execute_ttnn(
        test_with_basic_callables,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_zeros(
    shape: Shape, dtype: torch.dtype, request, device
):
    def test_with_zeros_init(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        builder.set_goldens({in0: torch.zeros})
        zeros_result = builder.neg(in0, unit_attrs=unit_attrs)
        return zeros_result

    compile_and_execute_ttnn(
        test_with_zeros_init,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_ones(
    shape: Shape, dtype: torch.dtype, request, device
):
    def test_with_ones_init(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        builder.set_goldens({in0: torch.ones})
        ones_result = builder.neg(in0, unit_attrs=unit_attrs)
        return ones_result

    compile_and_execute_ttnn(
        test_with_ones_init,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_eye(shape: Shape, dtype: torch.dtype, request, device):
    def test_with_eye_init(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        def eye_init(s):
            if len(s) == 2 and s[0] == s[1]:
                return torch.eye(s[0])
            elif len(s) == 2:
                return torch.eye(s[0], s[1])
            else:
                raise ValueError(f"torch.eye only supports 2D shapes, got {s}")

        builder.set_goldens({in0: eye_init})
        eye_result = builder.abs(in0, unit_attrs=unit_attrs)
        return eye_result

    compile_and_execute_ttnn(
        test_with_eye_init,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_mixed(
    shape: Shape, dtype: torch.dtype, request, device
):
    def test_with_mixed_init(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_goldens({in0: torch.zeros, in1: torch.ones})
        add_result = builder.add(in0, in1, unit_attrs=unit_attrs)
        return add_result

    compile_and_execute_ttnn(
        test_with_mixed_init,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_custom_lambda(
    shape: Shape, dtype: torch.dtype, request, device
):
    def test_with_custom_lambda(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        custom_init = lambda s: torch.full(s, 2.0)
        builder.set_goldens({in0: custom_init})
        result = builder.multiply(in0, in0, unit_attrs=unit_attrs)
        return result

    compile_and_execute_ttnn(
        test_with_custom_lambda,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_error_handling(shape: Shape, dtype: torch.dtype):
    """Test error handling for invalid callable initialization functions"""

    def test_with_invalid_callable(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        invalid_init = lambda s: "not a tensor"

        result = builder.neg(in0, unit_attrs=unit_attrs)
        with pytest.raises((TypeError, RuntimeError)):
            builder.set_goldens({in0: invalid_init})
        return result

    def test_with_failing_callable(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        failing_init = lambda s: torch.zeros(s) / 0  # Division by zero

        result = builder.neg(in0, unit_attrs=unit_attrs)
        with pytest.raises(RuntimeError):
            builder.set_goldens({in0: failing_init})
        return result


@pytest.mark.parametrize("shapes", [[(128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dim_arg", [[1]])
def test_argmax(shapes, dim_arg, request, device):
    def argmax(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.argmax(in0, dim_arg, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        argmax,
        inputs_shapes=shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.xfail(reason="`reverse` doesn't have a legalization. See issue #2495")
@pytest.mark.parametrize("shape", [(64, 64)], ids=shape_str)
@pytest.mark.parametrize("dims", [[0, 1]])
def test_reverse(shape: Shape, dims: List[int], request, device):
    def reverse(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reverse(in0, dims=dims, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        reverse,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip(reason="See issue #3685")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_and(shape: Shape, dim_args: List[int], request, device):
    def reduce_and(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reduce_and(in0, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        reduce_and,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def reduce_or(
    in0: Operand,
    builder: TTNNBuilder,
    dim_args: List[int],
    keep_dim: bool = False,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_or(
        in0, dim_args=dim_args, keep_dim=keep_dim, unit_attrs=unit_attrs
    )


@pytest.mark.xfail(reason="only floats are supported in runtime. See issue #1775")
@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_or(shape: Shape, dim_args: List[int], request, device):
    def reduce_or_wrapper(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return reduce_or(in0, builder, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        reduce_or_wrapper,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def permute(
    in0: Operand,
    builder: TTNNBuilder,
    permutation: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.permute(
        in0,
        permutation=permutation,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize("shapes", [[(2, 3, 4)]], ids=shapes_list_str)
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
def test_permute(shapes: List[Shape], permutation: List[int], request, device):
    # Create a wrapper function that captures permutation
    def permute_wrapper(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return permute(in0, builder, permutation, unit_attrs)

    # Set the name for better test identification
    permute_wrapper.__name__ = "permute"

    compile_and_execute_ttnn(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes", [[(10, 64, 32, 3), (10, 128, 128, 3)]], ids=shapes_list_str
)
@pytest.mark.parametrize("scale_factor", [[2, 4]])
def test_upsample2d(shapes: List[Shape], scale_factor: List[int], request, device):
    def upsample2d(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.upsample2d(
            in0,
            in1,
            scale_factor=scale_factor,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttnn(
        upsample2d,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape,start,end,step,dim", [((5,), 0, 5, 1, 0)])
def test_arange(
    shape: Shape, start: int, end: int, step: int, dim: int, request, device
):
    def arange(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.arange(in0, start, end, step, dim, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        arange,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "from_type,to_type",
    [
        pytest.param(
            torch.int32, torch.float32, marks=pytest.mark.xfail(reason="Golden failure")
        ),
        pytest.param(
            torch.float32, torch.int32, marks=pytest.mark.xfail(reason="Golden failure")
        ),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
    ids=["i32-f32", "f32-i32", "bf16-f32", "f32-bf16"],
)
def test_typecast(
    shape: Shape,
    from_type: torch.dtype,
    to_type: torch.dtype,
    request,
    device,
):
    def typecast(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.typecast(in0, output_type=to_type, unit_attrs=unit_attrs)

    pipeline_options = []
    compile_and_execute_ttnn(
        typecast,
        [shape],
        [from_type],
        test_base=request.node.name,
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shapes", [[(4, 4, 128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dim", [1])
def test_cumsum(shapes: List[Shape], dim: int, request, device):
    def cumsum(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.cumsum(in0, dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        cumsum,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def prod(in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.prod(in0, [1], False, unit_attrs=unit_attrs)


@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 3, 512)]], ids=shapes_list_str
)
def test_fill_cache(shapes: List[Shape], request, device):
    def fill_cache(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.fill_cache(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        fill_cache,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def softmax(
    in0: Operand,
    builder: TTNNBuilder,
    dimension: int = -1,
    numeric_stable: bool = False,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.softmax(
        in0, dimension=dimension, numeric_stable=numeric_stable, unit_attrs=unit_attrs
    )


@pytest.mark.parametrize("shape", [(512, 1024)], ids=shape_str)
@pytest.mark.parametrize("dimension", [-1])
@pytest.mark.parametrize("numeric_stable", [False, True])
def test_softmax(shape: Shape, dimension: int, numeric_stable: bool, request, device):
    # Create a wrapper function that captures dimension
    def softmax_wrapper(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return softmax(in0, builder, dimension, numeric_stable, unit_attrs)

    # Set the name for better test identification
    softmax_wrapper.__name__ = "softmax"

    compile_and_execute_ttnn(
        softmax_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.xfail(reason="run error")
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.float32, torch.float32, torch.int32]])
def test_update_cache(shapes: List[Shape], dtypes: List[torch.dtype], request, device):
    def update_cache(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        update_cache,
        shapes,
        inputs_types=dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def embedding(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.embedding(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.qint32,
        pytest.param(
            torch.qint8,
            marks=pytest.mark.skip(
                reason="qint8 quantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
    ids=["qint32", "qint8"],
)
def test_quantize(
    shape: Shape,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
    device,
):
    def quantize(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.quantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        quantize,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "input_dtype",
    [
        TypeInfo(torch.qint32, 0.1, 0),
        pytest.param(
            TypeInfo(torch.qint8, 0.1, 0),
            marks=pytest.mark.skip(
                reason="qint8 dequantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
    ids=["qint32", "qint8"],
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_dequantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
    device,
):
    def dequantize(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.dequantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        dequantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "input_dtype",
    [
        TypeInfo(torch.qint32, 0.1, 0),
        pytest.param(
            TypeInfo(torch.qint8, 0.1, 0),
            marks=pytest.mark.skip(
                reason="qint8 requantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.qint32,
        pytest.param(
            torch.qint8, marks=pytest.mark.skip(reason="qint8 quantize not supported")
        ),
    ],
    ids=["qint32", "qint8"],
)
def test_requantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
    device,
):
    def requantize(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.requantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        requantize,
        [shape],
        inputs_types=[input_dtype],
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


reduction_ops = [
    sum,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("test_fn", reduction_ops)
def test_reduction_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
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
        device=device,
        pipeline_options=pipeline_options,
    )


unary_ops_int32 = [
    pytest.param(
        sum,
        marks=pytest.mark.skip(
            reason="Sum does not support int32 input. Issue: https://github.com/tenstorrent/tt-metal/issues/26724"
        ),
    ),
    pytest.param(
        max,
        marks=pytest.mark.skip(
            reason="Max does not support int32 input. Issue: https://github.com/tenstorrent/tt-metal/issues/26726"
        ),
    ),
    pytest.param(
        min,
        marks=pytest.mark.skip(
            reason="Min does not support int32 input. Issue: https://github.com/tenstorrent/tt-metal/issues/26726"
        ),
    ),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
# TODO (anuragsingh): Add tt-metal and ttnn-standalone tests. Link to issue: https://github.com/tenstorrent/tt-mlir/issues/4444
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", unary_ops_int32)
def test_unary_ops_int32(
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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize(
    "test_fn",
    [
        matmul,
    ],
)
def test_matmul(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    # NOTE: this function is _only_ for binary ops that take the same shape arguments
    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize(
    "test_fn,inputs_shapes,inputs_dtypes",
    [
        (transpose, [(64, 32)], [torch.float32]),
        pytest.param(
            reshape,
            [(64, 32)],
            [torch.float32],
        ),
        pytest.param(
            embedding,
            [(33, 32), (512, 128)],
            [torch.float32] * 2,
        ),
    ],
)
def test_unique_ops(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_dtypes: List[torch.dtype],
    request,
    device,
):
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=inputs_shapes,
        inputs_types=inputs_dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


def slice(
    in0: Operand,
    begins: List[int],
    ends: List[int],
    step: Optional[List[int]] = None,
    builder: TTNNBuilder = None,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.slice(in0, begins, ends, step, unit_attrs=unit_attrs)


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
    def slice_op(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return slice(in0, begins, ends, step, builder, unit_attrs)

    compile_and_execute_ttnn(
        slice_op,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def gather(
    in0: Operand,
    builder: TTNNBuilder,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    indices_dtype: torch.dtype,
    unit_attrs: Optional[List[str]] = None,
):
    # For now, just create zero indices - this tests the basic gather functionality.
    # In a real test, you'd want to create varied indices to test different gather patterns.
    indices = builder.zeros(indices_shape, indices_dtype)

    # Set collapsed_slice_dims to be the same as start_index_map
    # This is what the GatherToEmbeddingConversionPattern expects.
    collapsed_slice_dims = start_index_map

    # Set remaining parameters to empty lists for simplicity.
    operand_batching_dims = []
    start_indices_batching_dims = []

    # Set index_vector_dim correctly based on the use case.
    if len(indices_shape) == 1 and len(start_index_map) == 1:
        # Single indices case - index vector dim is implicit.
        index_vector_dim = len(indices_shape)  # = 1
    else:
        # Multi-dimensional indices - last dimension contains index vectors.
        index_vector_dim = len(indices_shape) - 1

    return builder.gather(
        in0,
        indices,
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        operand_batching_dims=operand_batching_dims,
        start_indices_batching_dims=start_indices_batching_dims,
        start_index_map=start_index_map,
        index_vector_dim=index_vector_dim,
        slice_sizes=slice_sizes,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize(
    "input_shape,input_dtype,indices_shape,start_index_map,offset_dims,slice_sizes",
    [
        # Simple 1D indices - f32.
        ((100, 50), torch.float32, (10,), [0], [1], [1, 50]),
        pytest.param(
            (8, 16, 32),
            torch.float32,
            (4, 2, 2),
            [0, 2],
            [1],
            # Complex indices - f32.
            [1, 16, 1],
        ),
    ],
    ids=[
        "simple_1d-f32",
        "complex_indices-f32",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_gather(
    input_shape: Shape,
    input_dtype: torch.dtype,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    target: str,
    request,
    device,
):
    def gather_wrapper(in0: Operand, builder: TTNNBuilder):
        return gather(
            in0,
            builder,
            indices_shape,
            start_index_map,
            offset_dims,
            slice_sizes,
            input_dtype,
        )

    compile_and_execute_ttnn(
        gather_wrapper,
        [input_shape],
        [input_dtype],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((32, 128), [128]),
        ((2, 4, 64), [64]),
    ],
)
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_rms_norm(
    shape: Shape,
    normalized_shape: List[int],
    has_weight: bool,
    has_bias: bool,
    target: str,
    request,
    device,
):
    def rms_norm(*inputs, unit_attrs: Optional[List[str]] = None):

        builder = inputs[-1]
        # Extract inputs based on test configuration
        in0 = inputs[0]
        weight = None
        bias = None

        if has_weight and len(inputs) > 1:
            weight = inputs[1]
        if has_bias:
            if has_weight and len(inputs) > 2:
                bias = inputs[2]
            elif not has_weight and len(inputs) > 1:
                bias = inputs[1]

        return builder.rms_norm(
            in0,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            unit_attrs=unit_attrs,
        )

    # Determine input shapes
    shapes = [shape]
    if has_weight:
        shapes.append(tuple(normalized_shape))
    if has_bias:
        shapes.append(tuple(normalized_shape))

    compile_and_execute_ttnn(
        rms_norm,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize(
    "input_rank, shard_dims",
    [
        (5, (1, 4)),
        (5, (4, 1)),
        (5, (2, 4)),
        (5, (1, 4)),
        (5, (-1, 3)),
        (5, (4, -1)),
        (5, (-1, 4)),
        (5, (-1, 0)),
        (4, (1, 3)),
        (4, (3, 1)),
        (4, (2, 3)),
        (4, (3, 2)),
        (4, (0, 2)),
        (4, (1, 0)),
        (4, (-1, 3)),
        (4, (3, -1)),
        (4, (-1, 1)),
        (4, (1, -1)),
        (3, (1, 2)),
        (3, (2, 1)),
        (3, (0, 1)),
        (3, (1, 0)),
        (3, (-1, 2)),
        (3, (2, -1)),
        (3, (-1, 1)),
        (3, (0, -1)),
        (2, (0, 1)),
        (2, (1, 0)),
        (2, (-1, 1)),
        (2, (1, -1)),
        (2, (-1, 0)),
        (2, (0, -1)),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (4, 2), (1, 8), (8, 1), (1, 2), (2, 1)], ids=shape_str
)
def test_mesh_shard_devices(
    input_rank: int,
    shard_dims: Tuple[int, int],
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    shard_shape = make_shard_shape(input_rank, shard_dims, mesh_shape)
    if all(x == 1 for x in shard_shape):
        pytest.skip("sharding is meaningless, skipping test.")
    input_shape = [n_shards for idx, n_shards in enumerate(shard_shape)]

    def mesh_shard_devices(in0: Operand, builder: TTNNBuilder):
        mesh_shard_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        neg_output = builder.neg(mesh_shard_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )

    compile_and_execute_ttnn(
        mesh_shard_devices,
        [input_shape],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 32, 32, 32),
        (1, 32, 32, 1),
        (32, 32, 1, 1),
        (1, 32, 32),
        (32, 32),
        (32, 40),
        (40, 32),
        pytest.param((1, 1, 32, 32, 32), marks=pytest.mark.xfail(reason="run error")),
        pytest.param(
            (1, 1, 1, 1, 1, 1, 32, 32, 32), marks=pytest.mark.xfail(reason="run error")
        ),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("all_gather_dim", range(4))
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_gather(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if all_gather_dim >= len(test_shape):
        pytest.skip("all_gather_dim is out of range")
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_gather across 1 device is meaningless")

    def all_gather(mesh_shard_in: Operand, builder: TTNNBuilder):
        return builder.all_gather(
            mesh_shard_in,
            all_gather_dim=all_gather_dim,
            cluster_axis=cluster_axis,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, all_gather)

    compile_and_execute_ttnn(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        pytest.param((1, 1, 1, 256, 256), marks=pytest.mark.xfail(reason="run error")),
        (1, 1, 256, 256),
        (1, 1, 256, 257),
        (1, 1, 256, 255),
        (1, 256, 256, 1),
        (256, 256, 1, 1),
        (1, 1, 32, 64),
        (1, 64, 64),
        (64, 64),
        (64, 65),
        (32, 64),
        pytest.param(
            (33, 65), marks=pytest.mark.xfail(reason="run error")
        ),  # all_gather + local reduce case
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_reduce(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")

    # test 'sum' only for now. Other reduce types are not supported yet.
    def all_reduce(mesh_shard_in: Operand, builder: TTNNBuilder):
        return builder.all_reduce(
            mesh_shard_in,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=cluster_axis,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, all_reduce)

    compile_and_execute_ttnn(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 256, 256),
        (1, 1, 256, 257),
        (1, 1, 256, 255),
        (1, 256, 256, 1),
        (256, 256, 1, 1),
        (1, 1, 32, 64),
        (1, 128, 128),
        (128, 128),
        (128, 129),
        (64, 128),
        (64, 24),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("scatter_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_reduce_scatter(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    scatter_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")
    if scatter_dim >= len(test_shape):
        pytest.skip("scatter_dim is out of range")
    if test_shape[scatter_dim] % mesh_shape[cluster_axis] != 0:
        pytest.skip("scatter_dim is not divisible by mesh_shape[cluster_axis]")

    # test 'sum' only for now. Other reduce types are not supported yet.
    def reduce_scatter(mesh_shard_in: Operand, builder: TTNNBuilder):
        return builder.reduce_scatter(
            mesh_shard_in,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=scatter_dim,
            cluster_axis=cluster_axis,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, reduce_scatter)

    compile_and_execute_ttnn(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 32, 64),
        (1, 32, 64),
        (32, 64),
        (30, 60),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape, source_target_pairs",
    [
        pytest.param(
            (1, 2), [(0, 1)], marks=pytest.mark.xfail(reason="Fails Golden")
        ),  # https://github.com/tenstorrent/tt-mlir/issues/4323
        ((1, 2), [(0, 1), (1, 0)]),
        ((2, 4), [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ((2, 4), [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]),
        ((2, 4), [(0, 4), (4, 0), (1, 5), (5, 1), (2, 6), (6, 2), (3, 7), (7, 3)]),
        ((2, 4), [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)]),
        ((2, 4), [(0, 2), (1, 3), (4, 6), (5, 7), (2, 0), (3, 1), (6, 4), (7, 5)]),
        ((2, 4), [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]),
        pytest.param(
            (2, 4),
            [(0, 1), (2, 3), (4, 5), (6, 7)],
            marks=pytest.mark.xfail(
                reason="https://github.com/tenstorrent/tt-mlir/issues/4323"
            ),
        ),
        ((1, 8), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]),
        ((1, 32), [(i, (i + 1) % 32) for i in range(32)]),
        (
            (8, 4),
            # fmt: off
            # rotate right within each cluster along axis 1
            [
                (0, 1), (1, 2), (2, 3), (3, 0), # cluster #0
                (4, 5), (5, 6), (6, 7),( 7, 4), # cluster #1
                (8, 9), (9, 10), (10, 11), (11, 8), # cluster #2
                (12, 13), (13, 14), (14, 15), (15, 12), # cluster #3
                (16, 17), (17, 18), (18, 19), (19, 16), # cluster #4
                (20, 21), (21, 22), (22, 23), (23, 20), # cluster #5
                (24, 25), (25, 26), (26, 27), (27, 24), # cluster #6
                (28, 29), (29, 30), (30, 31), (31, 28), # cluster #7
            ],
            # fmt: on
        ),
        (
            (8, 4),
            # fmt: off
            # rotate right within each cluster along axis 0
            [
                (0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 0), # cluster #0
                (1, 5), (5, 9), (9, 13), (13, 17), (17, 21), (21, 25), (25, 29), (29, 1), # cluster #1
                (2, 6), (6, 10), (10, 14), (14, 18), (18, 22), (22, 26), (26, 30), (30, 2), # cluster #2
                (3, 7), (7, 11), (11, 15), (15, 19), (19, 23), (23, 27), (27, 31), (31, 3), # cluster #3
            ],
            # fmt: on
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_collective_permute(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    source_target_pairs: List[Tuple[int, int]],
    dtype: torch.dtype,
    request,
    device,
):
    max_id = reduce(operator.mul, mesh_shape, 1)
    if not all(pair[0] < max_id and pair[1] < max_id for pair in source_target_pairs):
        pytest.skip("Source and target pairs are out of range")

    def collective_permute(mesh_shard_in: Operand, builder: TTNNBuilder):
        return builder.collective_permute(
            mesh_shard_in,
            source_target_pairs=source_target_pairs,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, collective_permute)

    compile_and_execute_ttnn(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (32, 64),
        (32, 64, 128),
        (8, 8, 64, 64),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("split_dim", range(4))
@pytest.mark.parametrize("concat_dim", range(4))
@pytest.mark.parametrize(
    "mesh_shape, replica_groups",
    [
        ((1, 8), ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((2, 4), ((0, 4), (1, 5), (2, 6), (3, 7))),
        ((2, 4), ((0, 1, 2, 3), (4, 5, 6, 7))),
        ((4, 2), ((0, 2, 4, 6), (1, 3, 5, 7))),
        ((4, 2), ((0, 1), (2, 3), (4, 5), (6, 7))),
        ((1, 2), ((0, 1),)),
        ((2, 1), ((0, 1),)),
        ((1, 32), range(32)),
        (
            (8, 4),
            (
                (0, 1, 2, 3, 4, 5, 6, 7),
                (8, 9, 10, 11, 12, 13, 14, 15),
                (16, 17, 18, 19, 20, 21, 22, 23),
                (24, 25, 26, 27, 28, 29, 30, 31),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_to_all(
    test_shape: Shape,
    split_dim,
    concat_dim,
    mesh_shape,
    replica_groups,
    dtype: torch.dtype,
    request,
    device,
):
    split_count = len(replica_groups[0])
    if split_dim >= len(test_shape):
        pytest.skip("Split dimension is out of range")
    if concat_dim >= len(test_shape):
        pytest.skip("Concat dimension is out of range")

    def all_to_all(mesh_shard_in: Operand, builder: TTNNBuilder):
        return builder.all_to_all(
            mesh_shard_in,
            split_dim=split_dim,
            concat_dim=concat_dim,
            split_count=split_count,
            replica_groups=replica_groups,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, all_to_all)

    compile_and_execute_ttnn(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (64, 32),
        (32, 128, 64),
        (8, 8, 32, 64),
        (10, 10, 30, 60),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape, replica_groups",
    [
        ((2, 4), [(0, 1, 2, 3), (4, 5, 6, 7)]),
        ((2, 4), [(0, 4), (1, 5), (2, 6), (3, 7)]),
        ((4, 2), [(0, 1), (2, 3), (4, 5), (6, 7)]),
        ((4, 2), [(0, 2, 4, 6), (1, 3, 5, 7)]),
        ((1, 8), [(0, 1, 2, 3, 4, 5, 6, 7)]),
        ((1, 2), ((0, 1),)),
        ((2, 1), ((0, 1),)),
        ((1, 32), range(32)),
        (
            (8, 4),
            (
                (0, 1, 2, 3, 4, 5, 6, 7),
                (8, 9, 10, 11, 12, 13, 14, 15),
                (16, 17, 18, 19, 20, 21, 22, 23),
                (24, 25, 26, 27, 28, 29, 30, 31),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_collective_broadcast(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    replica_groups,
    dtype: torch.dtype,
    request,
    device,
):
    def collective_broadcast(mesh_shard_in: Operand, builder: TTNNBuilder):
        return builder.collective_broadcast(
            mesh_shard_in,
            replica_groups=replica_groups,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, collective_broadcast)

    compile_and_execute_ttnn(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
