# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional

from ttir_builder import Operand, TTIRBuilder, Shape, TypeInfo
from ttir_builder.utils import compile_to_flatbuffer, Marks, shape_str


def exp(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.exp(in0, unit_attrs=unit_attrs)


def expm1(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.expm1(in0, unit_attrs=unit_attrs)


def ceil(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.ceil(in0, unit_attrs=unit_attrs)


def floor(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.floor(in0, unit_attrs=unit_attrs)


def abs(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.abs(in0, unit_attrs=unit_attrs)


def logical_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.logical_not(in0, unit_attrs=unit_attrs)


def bitwise_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.bitwise_not(in0, unit_attrs=unit_attrs)


def neg(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)


def sign(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sign(in0, unit_attrs=unit_attrs)


def sin(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)


def cos(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cos(in0, unit_attrs=unit_attrs)


# Special handling for tan PCC checks. Due to the vertical asymptote on the tan graph, small changes in input values result in large changes in output values at multiples of pi/2, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_tan(shape: Shape, dtype: torch.dtype, request):
    def tan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        import math

        randn_tensor = torch.randn(shape, dtype=dtype)
        input_golden = randn_tensor.uniform_(
            (-math.pi / 2 + 0.02), (math.pi / 2 - 0.02)
        )
        output_golden = torch.tan(input_golden)
        builder.set_graph_input_output([input_golden], [output_golden], override=True)
        return builder.tan(in0, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        tan,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def atan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.atan(in0, unit_attrs=unit_attrs)


def tanh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)


# Special handling for log PCC checks. Due to the vertical asymptote on the log graph, small changes in input values result in large changes in output values at negative values, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_log(shape: Shape, dtype: torch.dtype, request):
    def log(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        randn_tensor = torch.randn(shape, dtype=dtype)
        abs_tensor = torch.abs(randn_tensor)
        error_margin = torch.full(randn_tensor.shape, 0.01)
        input_golden = torch.add(abs_tensor, error_margin)
        output_golden = torch.log(input_golden)
        builder.set_graph_input_output([input_golden], [output_golden], override=True)
        return builder.log(in0, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        log,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Special handling for log1p PCC checks. Due to the vertical asymptote on the log1p graph, small changes in input values result in large changes in output values at values below -1, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_log1p(shape: Shape, dtype: torch.dtype, request):
    def log1p(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        randn_tensor = torch.randn(shape, dtype=dtype)
        abs_tensor = torch.abs(randn_tensor)
        error_margin = torch.full(randn_tensor.shape, -0.99)
        input_golden = torch.add(abs_tensor, error_margin)
        output_golden = torch.log1p(input_golden)
        builder.set_graph_input_output([input_golden], [output_golden], override=True)
        return builder.log1p(in0, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        log1p,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def relu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)


def gelu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.gelu(in0, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request):
    def clamp_scalar(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.clamp_scalar(
            in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
        )

    compile_to_flatbuffer(
        clamp_scalar,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(32, 64), (32, 64), (32, 64), (32, 64)]])
def test_clamp_tensor(shapes: List[Shape], request):
    def clamp_tensor(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        in3: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.clamp_tensor(in0, in1, in2, in3, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        clamp_tensor,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def leaky_relu(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.leaky_relu(in0, unit_attrs=unit_attrs)


def sqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sqrt(in0, unit_attrs=unit_attrs)


def cbrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cbrt(in0, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_rsqrt(shape: Shape, dtype: torch.dtype, target: str, request):
    def rsqrt(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        input_tensor = torch.abs(torch.randn(shape, dtype=dtype))
        golden_output_tensor = torch.rsqrt(input_tensor)
        builder.set_graph_input_output(
            [input_tensor], [golden_output_tensor], override=True
        )
        return builder.rsqrt(in0, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        rsqrt,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def sigmoid(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)


def reciprocal(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.reciprocal(in0, unit_attrs=unit_attrs)


def is_finite(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.is_finite(in0, unit_attrs=unit_attrs)


def get_dimension_size(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.get_dimension_size(in0, unit_attrs=unit_attrs)


@pytest.mark.fails_golden
@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        (
            [(4, 10, 3, 5, 7), (4, 10, 5, 7, 3), (4, 10, 3, 7, 10, 7, 3)],
            [0],
            [3],
            [0],
            [2],
        )
    ],
)
def test_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    request,
):
    def dot_general(
        in0: Operand,
        in1: Operand,
        out0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.dot_general(
            in0,
            in1,
            out0,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        dot_general,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def add(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def multiply(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def logical_and(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def logical_or(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_or(in0, in1, unit_attrs=unit_attrs)


def logical_xor(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_xor(in0, in1, unit_attrs=unit_attrs)


def bitwise_and(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_and(in0, in1, unit_attrs=unit_attrs)


def bitwise_or(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_or(in0, in1, unit_attrs=unit_attrs)


def bitwise_xor(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_xor(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


def eq(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.eq(in0, in1, unit_attrs=unit_attrs)


def ne(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ne(in0, in1, unit_attrs=unit_attrs)


def ge(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ge(in0, in1, unit_attrs=unit_attrs)


def gt(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gt(in0, in1, unit_attrs=unit_attrs)


def le(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.le(in0, in1, unit_attrs=unit_attrs)


def lt(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.lt(in0, in1, unit_attrs=unit_attrs)


def div(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.div(in0, in1, unit_attrs=unit_attrs)


def remainder(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def maximum(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.maximum(in0, in1, unit_attrs=unit_attrs)


def minimum(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.minimum(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shapes", [[(10, 64, 32), (32, 128), (128,)]])
def test_linear(shapes: List[Shape], request):
    def linear(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.linear(in0, in1, in2, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        linear,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_pow(shape: Shape, dtype: torch.dtype, target: str, request):
    def pow(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        randn_base_tensor = torch.randn(shape, dtype=dtype)
        randn_exponent_tensor = torch.randn(shape, dtype=dtype)
        if torch.is_floating_point(randn_exponent_tensor):
            randn_base_tensor = torch.abs(randn_base_tensor)
        output_golden = torch.pow(randn_base_tensor, randn_exponent_tensor)
        builder.set_graph_input_output(
            [randn_base_tensor, randn_exponent_tensor], [output_golden], override=True
        )
        return builder.pow(in0, in1, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        pow,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def matmul(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.matmul(in0, in1, unit_attrs=unit_attrs)


def sum(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sum(in0, unit_attrs=unit_attrs)


def mean(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mean(in0, unit_attrs=unit_attrs)


def max(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.max(in0, unit_attrs=unit_attrs)


def min(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.min(in0, unit_attrs=unit_attrs)


def reshape(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    # Calculate total elements in the input tensor
    input_shape = builder.get_shape(in0)
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim

    # Reshape to a 1D tensor with all elements
    new_shape = [int(total_elements)]  # This must be a list of integers
    return builder.reshape(in0, new_shape, unit_attrs=unit_attrs)


def transpose(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.transpose(in0, unit_attrs=unit_attrs)


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dim_arg", [0])
@pytest.mark.parametrize("keep_dim", [False])
def test_prod(shape: Shape, dim_arg: int, keep_dim: bool, request):
    def prod(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.prod(in0, [dim_arg], keep_dim, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        prod,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def where(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.where(in0, in1, in2, unit_attrs=unit_attrs)


def broadcast(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    broadcast_dimensions: Optional[List[int]] = None,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.broadcast(
        in0, in1, broadcast_dimensions=broadcast_dimensions, unit_attrs=unit_attrs
    )


def concat(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    dim: int,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shapes", [[(1, 1, 32), (1, 16, 32)]])
@pytest.mark.parametrize("broadcast_dimensions", [[1, 16, 1]])
def test_broadcast(shapes: List[Shape], broadcast_dimensions: List[int], request):
    # Create a wrapper function that captures broadcast_dimensions
    def broadcast_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return broadcast(in0, in1, builder, broadcast_dimensions, unit_attrs)

    # Set the name for better test identification
    broadcast_wrapper.__name__ = "broadcast"

    compile_to_flatbuffer(
        broadcast_wrapper,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 128, 128, 1)])
@pytest.mark.parametrize("dim", [0])
def test_squeeze(shape: Shape, dim: int, request):
    def squeeze(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.squeeze(in0, dim, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        squeeze,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dim", [0])
def test_unsqueeze(shape: Shape, dim: int, request):
    def unsqueeze(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.unsqueeze(in0, dim, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        unsqueeze,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 32, 32)])
@pytest.mark.parametrize("dims", [[32, 1, 1]])
def test_repeat(shape: Shape, dims: List[int], request):
    def repeat(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.repeat(in0, dims=dims, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        repeat,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 8, 1, 12, 64),
            (1, 8, 1, 12, 64),
        ]
    ],
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
def test_repeat_interleave(shapes: List[Shape], repeats: int, dim: int, request):
    def repeat_interleave(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.repeat_interleave(
            in0, in1, repeats=repeats, dim=dim, unit_attrs=unit_attrs
        )

    compile_to_flatbuffer(
        repeat_interleave,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
)
@pytest.mark.parametrize("dim", [0])
def test_concat(shapes: List[Shape], dim: int, request):
    # Create a wrapper function that captures dim
    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return concat(in0, in1, in2, dim, builder, unit_attrs)

    # Set the name for better test identification
    concat_wrapper.__name__ = "concat"

    compile_to_flatbuffer(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
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
            (1, 16, 28, 64),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
def test_conv2d(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
):
    def conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        conv2d,
        shapes,
        dtypes,
        test_base=request.node.name,
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
            (1, 16, 28, 64),
        ]
    ],
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
):
    def conv2d_consteval(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        conv2d_consteval,
        shapes,
        argument_types_string="conv2d_consteval=input,parameter,parameter,parameter",
        test_base=request.node.name,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (3, 8, 8, 256),
            (256, 256, 3, 3),
            (1, 1, 1, 256),
            (1, 10, 10, 256),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
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
):
    def conv_transpose2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv_transpose2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        conv_transpose2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "kernel_height,kernel_width,stride_height,stride_width,dilation_height,dilation_width,ceil_mode,padding_left,padding_right,padding_top, padding_bottom",
    [(2, 2, 2, 2, 1, 1, False, 0, 0, 0, 0)],
)
@pytest.mark.parametrize("shapes", [[(1, 128, 128, 32), (1, 64, 64, 32)]])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2])
def test_max_pool2d(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    kernel_height: int,
    kernel_width: int,
    stride_height: int,
    stride_width: int,
    dilation_height: int,
    dilation_width: int,
    ceil_mode: bool,
    padding_left: int,
    padding_right: int,
    padding_top: int,
    padding_bottom: int,
    request,
):
    def max_pool2d(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.max_pool2d(
            in0,
            in1,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            stride_height=stride_height,
            stride_width=stride_width,
            dilation_height=dilation_height,
            dilation_width=dilation_width,
            ceil_mode=ceil_mode,
            padding_left=padding_left,
            padding_right=padding_right,
            padding_top=padding_top,
            padding_bottom=padding_bottom,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        max_pool2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shapes", [[(1, 1, 5, 5), (2, 6, 14, 18)]])
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
def test_pad(shapes: List[Shape], padding: List[int], value: int, request):
    def pad(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.pad(
            in0, in1, padding=padding, value=value, unit_attrs=unit_attrs
        )

    compile_to_flatbuffer(
        pad,
        inputs_shapes=shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(32, 64)])
@pytest.mark.parametrize("dim,begin,end,step", [(0, 0, 3, 1)])
def test_index(shape: Shape, dim: int, begin: int, end: int, step: int, request):
    def index(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.index(
            in0, dim=dim, begin=begin, end=end, step=step, unit_attrs=unit_attrs
        )

    compile_to_flatbuffer(
        index,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim,begin,length,stride", [(1, 2, 2, 2)])
def test_select(shape: Shape, dim: int, begin: int, length: int, stride: int, request):
    def select(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.select(
            in0,
            dim=dim,
            begin=begin,
            length=length,
            stride=stride,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        select,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# TODO: these three nullary tensor creation ops can probably be combined in some way
@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
def test_zeros(shape: Shape, request):
    def zeros(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.zeros(shape, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        zeros,
        inputs_shapes=[],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
def test_ones(shape: Shape, request):
    def ones(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.ones(shape, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        ones,
        inputs_shapes=[],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(128, 128)]])
@pytest.mark.parametrize("dim_arg", [[1]])
def test_argmax(shapes, dim_arg, request):
    def argmax(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.argmax(in0, dim_arg, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        argmax,
        inputs_shapes=shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip("`reverse` doesn't have a legalization. See issue #2495")
@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("dims", [[0, 1]])
def test_reverse(shape: Shape, dims: List[int], request):
    def reverse(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reverse(in0, dims=dims, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        reverse,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip("See issue #3685")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_and(shape: Shape, dim_args: List[int], request):
    def reduce_and(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reduce_and(in0, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        reduce_and,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip("Run error")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_or(shape: Shape, dim_args: List[int], request):
    def reduce_or(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reduce_or(in0, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        reduce_or,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def permute(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    permutation: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.permute(
        in0,
        in1,
        permutation=permutation,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize("shapes", [[(2, 3, 4), (3, 4, 2)]])
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
def test_permute(shapes: List[Shape], permutation: List[int], request):
    # Create a wrapper function that captures permutation
    def permute_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return permute(in0, in1, builder, permutation, unit_attrs)

    # Set the name for better test identification
    permute_wrapper.__name__ = "permute"

    compile_to_flatbuffer(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(10, 64, 32, 3), (10, 128, 128, 3)]])
@pytest.mark.parametrize("scale_factor", [[2, 4]])
def test_upsample2d(shapes: List[Shape], scale_factor: List[int], request):
    def upsample2d(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.upsample2d(
            in0,
            in1,
            scale_factor=scale_factor,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        upsample2d,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape,start,end,step,dim", [((5,), 0, 5, 1, 0)])
def test_arange(shape: Shape, start: int, end: int, step: int, dim: int, request):
    def arange(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.arange(in0, start, end, step, dim, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        arange,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize(
    "from_type,to_type", [(torch.int32, torch.float32)], ids=["i32-f32"]
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_typecast(
    shape: Shape, from_type: torch.dtype, to_type: torch.dtype, target: str, request
):
    def typecast(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.typecast(in0, in1, unit_attrs=unit_attrs)

    pipeline_options = []
    # Workaround for ttmetal, only support 1x1 grid atm
    if target == "ttmetal":
        pipeline_options.append("override-device-shape=1,1")
    compile_to_flatbuffer(
        typecast,
        [shape, shape],
        [from_type, to_type],
        test_base=request.node.name,
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shapes", [[(4, 4, 128, 128), (4, 4, 128, 128)]])
@pytest.mark.parametrize("dim", [1])
def test_cumsum(shapes: List[Shape], dim: int, request):
    def cumsum(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.cumsum(in0, in1, dim=dim, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        cumsum,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def prod(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.prod(in0, [1], False, unit_attrs=unit_attrs)


@pytest.mark.fails_golden
@pytest.mark.parametrize("shapes", [[(1, 32, 64, 512), (1, 32, 3, 512)]])
def test_fill_cache(shapes: List[Shape], request):
    def fill_cache(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.fill_cache(in0, in1, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        fill_cache,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def softmax(
    in0: Operand,
    builder: TTIRBuilder,
    dimension: int = -1,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.softmax(in0, dimension=dimension, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(512, 1024)])
@pytest.mark.parametrize("dimension", [-1])
def test_softmax(shape: Shape, dimension: int, request):
    # Create a wrapper function that captures dimension
    def softmax_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return softmax(in0, builder, dimension, unit_attrs)

    # Set the name for better test identification
    softmax_wrapper.__name__ = "softmax"

    compile_to_flatbuffer(
        softmax_wrapper,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.run_error
@pytest.mark.parametrize("shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]])
@pytest.mark.parametrize("dtypes", [[torch.float32, torch.float32, torch.int32]])
def test_update_cache(shapes: List[Shape], dtypes: List[torch.dtype], request):
    def update_cache(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        update_cache,
        shapes,
        inputs_types=dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def embedding(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.embedding(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.qint32])
def test_quantize(
    shape: Shape, scale: float, zero_point: int, dtype: torch.dtype, request
):
    def quantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.quantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    pipeline_options = ["enable-const-eval=false"]  # temporary workaround. Issue #3505.
    compile_to_flatbuffer(
        quantize,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("input_dtype", [TypeInfo(torch.qint32, 0.1, 0)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_dequantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
):
    def dequantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.dequantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    pipeline_options = ["enable-const-eval=false"]  # temporary workaround. Issue #3505.
    compile_to_flatbuffer(
        dequantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("input_dtype", [TypeInfo(torch.qint32, 0.1, 0)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.qint32])
def test_requantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
):
    def requantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.requantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    pipeline_options = ["enable-const-eval=false"]  # temporary workaround. Issue #3505.
    compile_to_flatbuffer(
        requantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=pipeline_options,
    )


# Create hoisted versions of operations by currying the unit_attrs parameter
def create_hoisted_unary_op(op_func, name):
    """Create a hoisted version of a unary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, builder, **kwargs):
        # For unary ops
        return op_func(in0, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    # Set the name for better test identification
    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_binary_op(op_func, name):
    """Create a hoisted version of a binary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, in1, builder, **kwargs):
        return op_func(in0, in1, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_permute_op(op_func, name):
    """Create a hoisted version of the permute operation that calculates appropriate permutation dimensions"""

    def hoisted_op(in0, in1, builder, **kwargs):
        # Calculate appropriate permutation based on input dimensions
        input_shape = builder.get_shape(in0)
        ndims = len(input_shape)

        # Create a simple permutation that reverses the dimensions
        # This is guaranteed to be valid for any tensor
        permutation = list(range(ndims))
        permutation.reverse()

        return op_func(
            in0, in1, builder, permutation, unit_attrs=["ttir.should_hoist"], **kwargs
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_softmax_op(op_func, name):
    """Create a hoisted version of the softmax operation"""

    def hoisted_op(in0, builder, **kwargs):
        # Default dimension for the hoisted version (last dimension)
        default_dimension = -1
        return op_func(
            in0,
            builder,
            dimension=default_dimension,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_concat_op(op_func, name):
    """Create a hoisted version of the concat operation"""

    def hoisted_op(in0, in1, in2, builder, **kwargs):
        # Default dimension for the hoisted version (dimension 0)
        default_dim = 0
        return op_func(
            in0,
            in1,
            in2,
            default_dim,
            builder,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create hoisted versions of all hoistable operations with proper names
hoisted_unary_ops = [
    create_hoisted_unary_op(exp, "exp"),
    create_hoisted_unary_op(sqrt, "sqrt"),
    create_hoisted_unary_op(abs, "abs"),
    create_hoisted_unary_op(ceil, "ceil"),
    create_hoisted_unary_op(floor, "floor"),
    create_hoisted_unary_op(tanh, "tanh"),
    create_hoisted_unary_op(reciprocal, "reciprocal"),
    create_hoisted_unary_op(neg, "neg"),
    pytest.param(
        create_hoisted_unary_op(reshape, "reshape"),
        marks=pytest.mark.xfail(reason="Reshape does not lower to loops properly"),
    ),
    pytest.param(
        create_hoisted_unary_op(reshape, "reshape"),
        marks=pytest.mark.xfail(reason="Reshape not compiling properly"),
    ),
    create_hoisted_unary_op(transpose, "transpose"),
]

hoisted_binary_ops = [
    create_hoisted_binary_op(add, "add"),
    create_hoisted_binary_op(multiply, "multiply"),
    create_hoisted_binary_op(subtract, "subtract"),
    create_hoisted_binary_op(div, "div"),
]

hoisted_ternary_ops = [
    create_hoisted_concat_op(concat, "concat"),
]


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("test_fn", hoisted_unary_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_unary_ops(
    test_fn: Callable,
    shape: Shape,
    request,
    target: str,
    dtype: torch.dtype = torch.float32,
):
    """Test unary ops that support CPU hoisting"""
    compile_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=f"{request.node.name}",
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(128, 128), (128, 128)],  # Same shapes
        [(128, 128), (1, 128)],  # Broadcasting second dimension
        [(128, 128), (128, 1)],  # Broadcasting first dimension
        [(128, 128, 64), (128, 1, 64)],  # 3D tensors with broadcasting
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("test_fn", hoisted_binary_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_binary_ops(
    test_fn: Callable, shapes: List[Shape], dtype: torch.dtype, request, target: str
):
    """Test binary ops that support CPU hoisting"""
    compile_to_flatbuffer(
        test_fn,
        shapes,
        [dtype] * len(shapes),
        test_base=f"{request.node.name}",
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Test hoisted permute separately because it requires unique input shapes.
@pytest.mark.parametrize(
    "shapes_and_perms",
    [
        # [(input_shape, output_shape), permutation]
        [[(2, 3, 4), (4, 2, 3)], [2, 0, 1]],
        [[(128, 128), (128, 128)], [0, 1]],
        [[(128, 64, 32), (32, 128, 64)], [2, 0, 1]],
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.fails_golden
def test_hoisted_permute(shapes_and_perms, request, target: str):
    shapes, permutation = shapes_and_perms

    def permute_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return permute(in0, in1, builder, permutation, unit_attrs=["ttir.should_hoist"])

    permute_wrapper.__name__ = "hoisted_permute"

    compile_to_flatbuffer(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


unary_ops = [
    exp,
    expm1 | Marks(pytest.mark.skip_target("ttmetal")),
    floor | Marks(pytest.mark.skip_target("ttmetal")),
    abs | Marks(pytest.mark.skip_target("ttmetal")),
    logical_not | Marks(pytest.mark.skip_target("ttmetal")),
    neg,
    sign | Marks(pytest.mark.skip_target("ttmetal")),
    cos,
    sin,
    atan | Marks(pytest.mark.skip_target("ttmetal")),
    tanh | Marks(pytest.mark.skip_target("ttmetal")),
    relu | Marks(pytest.mark.skip_target("ttmetal")),
    gelu | Marks(pytest.mark.skip_target("ttmetal")),
    leaky_relu | Marks(pytest.mark.skip_target("ttmetal")),
    sqrt | Marks(pytest.mark.skip_target("ttmetal")),
    cbrt | Marks(pytest.mark.skip_target("ttmetal")),
    sigmoid | Marks(pytest.mark.fails_golden),
    reciprocal | Marks(pytest.mark.skip_target("ttmetal")),
    is_finite | Marks(pytest.mark.skip_target("ttmetal")),
    ceil | Marks(pytest.mark.fails_golden),
    sum | Marks(pytest.mark.skip_target("ttmetal")),
    mean | Marks(pytest.mark.skip_target("ttmetal")),
    max | Marks(pytest.mark.fails_golden, pytest.mark.skip_target("ttmetal")),
    min | Marks(pytest.mark.fails_golden, pytest.mark.skip_target("ttmetal")),
    get_dimension_size | Marks(pytest.mark.skip_target("ttmetal")),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
):
    pipeline_options = []
    compile_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        add,
        multiply,
        subtract,
        eq | Marks(pytest.mark.skip_target("ttmetal")),
        ne | Marks(pytest.mark.skip_target("ttmetal")),
        le | Marks(pytest.mark.skip_target("ttmetal")),
        lt | Marks(pytest.mark.skip_target("ttmetal")),
        ge | Marks(pytest.mark.skip_target("ttmetal")),
        gt | Marks(pytest.mark.skip_target("ttmetal")),
        div | Marks(pytest.mark.skip_target("ttmetal")),
        remainder | Marks(pytest.mark.skip_target("ttmetal")),
        maximum,
        minimum | Marks(pytest.mark.skip_target("ttmetal")),
        matmul | Marks(pytest.mark.skip_target("ttmetal")),
        logical_and | Marks(pytest.mark.skip_target("ttmetal")),
        logical_or | Marks(pytest.mark.skip_target("ttmetal")),
        logical_xor | Marks(pytest.mark.skip_target("ttmetal")),
    ],
)
def test_binary_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    # NOTE: this function is _only_ for binary ops that take the same shape arguments
    pipeline_options = []
    compile_to_flatbuffer(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


@pytest.mark.run_error
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("test_fn", [bitwise_and, bitwise_or, bitwise_xor])
def test_bitwise_binary_ops(test_fn: Callable, shape: Shape, request):
    compile_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape] * 2,
        inputs_types=[torch.int8] * 2,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_fn,inputs_shapes,inputs_dtypes",
    [
        (transpose, [(64, 32)], None),
        (reshape, [(64, 32)], None),
        pytest.param(
            embedding,
            [(33, 32), (512, 128)],
            [torch.float32] * 2,
        ),
        pytest.param(
            where,
            [(64, 64)] * 3,
            [torch.float32, torch.float32, torch.float32],
        ),
    ],
)
def test_unique_ops(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_dtypes: List[torch.dtype],
    request,
):
    compile_to_flatbuffer(
        test_fn,
        inputs_shapes=inputs_shapes,
        inputs_types=inputs_dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
