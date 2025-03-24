# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import pytest
import torch
from typing import Callable, List

from ttmlir.test_utils import compile_to_flatbuffer
from ttmlir.ttir_builder import Operand, TTIRBuilder, UnitAttr, Shape, TypeInfo
from ttmlir.dialects import ttir
from ttmlir.passes import GoldenTensor, DataType
from ttmlir.ir import (
    DenseI64ArrayAttr,
    DenseI32ArrayAttr,
    IntegerAttr,
    IntegerType,
)

def exp(in0: Operand, builder: TTIRBuilder):
    return builder.exp(in0)


def expm1(in0: Operand, builder: TTIRBuilder):
    return builder.expm1(in0)


def ceil(in0: Operand, builder: TTIRBuilder):
    return builder.ceil(in0)


def floor(in0: Operand, builder: TTIRBuilder):
    return builder.floor(in0)


def abs(in0: Operand, builder: TTIRBuilder):
    return builder.abs(in0)


def logical_not(in0: Operand, builder: TTIRBuilder):
    return builder.logical_not(in0)


def bitwise_not(in0: Operand, builder: TTIRBuilder):
    return builder.bitwise_not(in0)


def neg(in0: Operand, builder: TTIRBuilder):
    return builder.neg(in0)


def sign(in0: Operand, builder: TTIRBuilder):
    return builder.sign(in0)


def sin(in0: Operand, builder: TTIRBuilder):
    return builder.sin(in0)


def cos(in0: Operand, builder: TTIRBuilder):
    return builder.cos(in0)


def tan(in0: Operand, builder: TTIRBuilder):
    return builder.tan(in0)


def atan(in0: Operand, builder: TTIRBuilder):
    return builder.atan(in0)


def tanh(in0: Operand, builder: TTIRBuilder):
    return builder.tanh(in0)


def log(in0: Operand, builder: TTIRBuilder):
    return builder.log(in0)


def log1p(in0: Operand, builder: TTIRBuilder):
    return builder.log1p(in0)


def relu(in0: Operand, builder: TTIRBuilder):
    return builder.relu(in0)


def gelu(in0: Operand, builder: TTIRBuilder):
    return builder.gelu(in0)


@pytest.mark.parametrize("shape", [[(64, 128)]])
@pytest.mark.parametrize("max_arg,min_arg", [[3.0, 2.0]])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request):
    def clamp_scalar(in0: Operand, builder: TTIRBuilder):
        return builder.clamp_scalar(in0, max_arg=max_arg, min_arg=min_arg)
    compile_to_flatbuffer(clamp_scalar, [shape], test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(32, 64), (32, 64), (32, 64), (32, 64)]])
def test_clamp_tensor(shapes: List[Shape], request):
    def clamp_tensor(in0: Operand, in1: Operand, in2: Operand, in3: Operand, builder: TTIRBuilder):
        return builder.clamp_tensor(in0, in1, in2, in3)
    compile_to_flatbuffer(clamp_tensor, shapes, test_base=request.node.name)


def leaky_relu(in0: Operand, builder: TTIRBuilder):
    return builder.leaky_relu(in0)


def sqrt(in0: Operand, builder: TTIRBuilder):
    return builder.sqrt(in0)


def cbrt(in0: Operand, builder: TTIRBuilder):
    return builder.cbrt(in0)


def rsqrt(in0: Operand, builder: TTIRBuilder):
    return builder.rsqrt(in0)


def sigmoid(in0: Operand, builder: TTIRBuilder):
    return builder.sigmoid(in0)


def reciprocal(in0: Operand, builder: TTIRBuilder):
    return builder.reciprocal(in0)


def is_finite(in0: Operand, builder: TTIRBuilder):
    return builder.is_finite(in0)


def get_dimension_size(in0: Operand, builder: TTIRBuilder):
    return builder.get_dimension_size(in0)

@pytest.mark.parametrize("shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
                         [ (
    [(4, 10, 3, 5, 7), (4, 10, 5, 7, 3)],
    [0], [3], [0], [2])])
def test_dot_general(shapes: List[Shape], batch_dims_lhs: List[int],
                     contract_dims_lhs: List[int], batch_dims_rhs: List[int],
                     contract_dims_rhs: List[int], request):
    def dot_general(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.dot_general(in0, in1, batch_dims_lhs, contract_dims_lhs, batch_dims_rhs, contract_dims_rhs) 
    compile_to_flatbuffer(dot_general, shapes, test_base=request.node.name)

def add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)


def multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.multiply(in0, in1)


def logical_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_and(in0, in1)


def logical_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_or(in0, in1)


def logical_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_xor(in0, in1)


def bitwise_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_and(in0, in1)


def bitwise_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_or(in0, in1)


def bitwise_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_xor(in0, in1)


def subtract(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.subtract(in0, in1)


def eq(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.eq(in0, in1)


def ne(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ne(in0, in1)


def ge(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ge(in0, in1)


def gt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.gt(in0, in1)


def le(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.le(in0, in1)


def lt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.lt(in0, in1)


def div(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.div(in0, in1)


def remainder(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.remainder(in0, in1)


def maximum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.maximum(in0, in1)


def minimum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.minimum(in0, in1)


@pytest.mark.parametrize("shapes", [[(10, 64, 32), (32, 128), (128,)]])
def test_linear(shapes: List[Shape], request):
    def linear(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        return builder.linear(in0, in1, in2)
    compile_to_flatbuffer(linear, shapes, test_base=request.node.name)

def pow(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.pow(in0, in1)


def matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.matmul(in0, in1)


def sum(in0: Operand, builder: TTIRBuilder):
    return builder.sum(in0)


def mean(in0: Operand, builder: TTIRBuilder):
    return builder.mean(in0)

def max(in0: Operand, builder: TTIRBuilder):
    return builder.max(in0)


def min(in0: Operand, builder: TTIRBuilder):
    return builder.min(in0)


def reshape(in0: Operand, builder: TTIRBuilder):
    return builder.reshape(in0, [2048])


def transpose(in0: Operand, builder: TTIRBuilder):
    return builder.transpose(in0)


def where(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.where(in0, in1, in2)


@pytest.mark.parametrize("shapes", [[ (64, 128), (32, 128), (16, 128), ]])
@pytest.mark.parametrize("dim", [0])
def test_concat( shapes: List[Shape], dim: int, request):
    def concat(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        return builder.concat([in0, in1, in2], dim=dim)
    compile_to_flatbuffer(concat, shapes, test_base=request.node.name)

@pytest.mark.skip("This test is not valid for TTRT Perf due to weird issues with perf collection. Issue #2371")
@pytest.mark.parametrize("shape", [(1, 128, 128, 1)])
@pytest.mark.parametrize("dim", [0])
def test_squeeze(shape: Shape, dim: int, request):
    def squeeze(in0: Operand, builder: TTIRBuilder):
        return builder.squeeze(in0, dim)
    compile_to_flatbuffer(squeeze, [shape], test_base=request.node.name)


# NOTE: Same as Squeeze, this Op is not valid for TTRT Perf. Issue #2371
@pytest.mark.skip("This test is not valid for TTRT Perf due to weird issues with perf collection. Issue #2371")
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dim", [0])
def test_unsqueeze(shape: Shape, dim: int, request):
    def unsqueeze(in0: Operand, builder: TTIRBuilder):
        return builder.unsqueeze(in0, dim)
    compile_to_flatbuffer(unsqueeze, [shape], test_base=request.node.name)

@pytest.mark.parametrize("shape", [(1, 32, 32)])
@pytest.mark.parametrize("dims", [[32, 1, 1]])
def test_repeat(shape: Shape, dims: List[int], request):
    def repeat(in0: Operand, builder: TTIRBuilder):
        return builder.repeat(in0, dims=dims)
    compile_to_flatbuffer(repeat, [shape], test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(1, 8, 1, 12, 64), (1, 8, 1, 12, 64),],])
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [2])
def test_repeat_interleave(shapes: List[Shape], repeats: int, dim: int):
    def repeat_interleave(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.repeat_interleave(in0, in1, repeats=repeats, dim=dim)
    compile_to_flatbuffer(repeat_interleave, shapes, test_base="test_repeat_interleave")


@pytest.mark.parametrize("shapes", [[(1, 1, 32), (1, 16, 32)]])
@pytest.mark.parametrize("broadcast_dimensions", [[1, 16, 1]])
def test_broadcast(shapes: List[Shape], broadcast_dimensions: List[int], request):
    def broadcast(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.broadcast(in0, in1, broadcast_dimensions=broadcast_dimensions)
    compile_to_flatbuffer(broadcast, shapes, test_base=request.node.name)


@pytest.mark.parametrize("shapes", [
    [
        (1, 32, 32, 64),
        (64, 32, 3, 3),
        (1, 1, 1, 64),
        (1, 16, 28, 64),
        ]
    ])
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("_stride,_padding,_dilation,_groups", [([2, 1], [2, 1], [2, 1], 2)])
def test_conv2d(
        shapes: List[Shape],
        dtypes: List[torch.dtype],
        _stride: List[int],
        _padding: List[int],
        _dilation: List[int],
        _groups: int,
        request):
    def conv2d(
        in0: Operand, weight: Operand, bias: Operand, in1: Operand, builder: TTIRBuilder
    ):
        #wrap attrs in attr types
        stride = DenseI32ArrayAttr.get(_stride)
        padding = DenseI32ArrayAttr.get(_padding)
        dilation = DenseI32ArrayAttr.get(_dilation)
        groups = IntegerAttr.get(IntegerType.get_signless(32), _groups)

        return builder.conv2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    compile_to_flatbuffer(conv2d, shapes, dtypes, test_base=request.node.name)


@pytest.mark.parametrize("shapes",[
    [
        (1, 32, 32, 64),
        (64, 32, 3, 3),
        (1, 1, 1, 64),
        (1, 16, 28, 64),
    ],
    inputs_types=[torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
    argument_types_string="test_conv2d_consteval=input,parameter,parameter,parameter",
)
def test_conv2d_consteval(
    in0: Operand, weight: Operand, bias: Operand, in1: Operand, builder: TTIRBuilder
):
    stride = DenseI32ArrayAttr.get([2, 1])
    padding = DenseI32ArrayAttr.get([2, 1])
    dilation = DenseI32ArrayAttr.get([2, 1])
    return builder.conv2d(
        in0,
        weight,
        bias,
        in1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=2,
    )


@compile_to_flatbuffer(
    [
        (3, 8, 8, 256),
        (256, 256, 3, 3),
        (1, 1, 1, 256),
        (1, 10, 10, 256),
    ]])
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("_stride,_padding,_output_padding,_dilation,_groups", [(1, 0, 0, 1, 1)])
def test_conv_transpose2d(
        shapes: List[Shape],
        dtypes: List[torch.dtype],
        _stride: int,
        _padding: int,
        _output_padding: int,
        _dilation: int,
        _groups: int,
        request):

    def conv_transpose2d(
        in0: Operand, weight: Operand, bias: Operand, in1: Operand, builder: TTIRBuilder
    ):
        #wrap attrs in attr types
        stride = IntegerAttr.get(IntegerType.get_signless(32), _stride)
        padding = IntegerAttr.get(IntegerType.get_signless(32), _padding)
        output_padding = IntegerAttr.get(IntegerType.get_signless(32), _output_padding)
        dilation = IntegerAttr.get(IntegerType.get_signless(32), _dilation)
        groups = IntegerAttr.get(IntegerType.get_signless(32), _groups)
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
        )
    compile_to_flatbuffer(conv_transpose2d, shapes, dtypes, test_base=request.node.name)


@pytest.mark.parametrize("kernel_height,kernel_width,stride_height,stride_width,dilation_height,dilation_width,ceil_mode,padding_left,padding_right,padding_top, padding_bottom", [(2,2,2,2,1,1,False,0,0,0,0)])
@pytest.mark.parametrize("shapes", [[(1, 128, 128, 32), (1, 64, 64, 32)]])
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 2])
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
        request
        ):

    def max_pool2d(in0: Operand, in1: Operand, builder: TTIRBuilder):
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
        )
    compile_to_flatbuffer(max_pool2d, shapes, dtypes, test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(1, 1, 5, 5), (2, 6, 14, 18)]])
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
def test_pad(shapes: List[Shape], padding: List[int], value: int, request):
    def pad(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.pad(in0, in1, padding=padding, value=value)
    compile_to_flatbuffer(pad, inputs_shapes=shapes, test_base=request.node.name)


@pytest.mark.parametrize("shape", [(32, 64)])
@pytest.mark.parametrize("dim,begin,end,step", [(0, 0, 3, 1)])
def test_index(shape: Shape, dim: int, begin: int, end: int, step: int, request):
    def index(in0: Operand, builder: TTIRBuilder):
        return builder.index(in0, dim=dim, begin=begin, end=end, step=step)
    compile_to_flatbuffer(index, [shape], test_base=request.node.name)


@pytest.mark.skip("`select` throwing floating point exception. See issue #2496")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim,begin,length", [(1,2,2)])
def test_select(shape: Shape, dim: int, begin: int, length: int, request):
    def select(in0: Operand, builder: TTIRBuilder):
        return builder.select(in0, dim = dim, begin = begin, length = length)
    compile_to_flatbuffer(select, [shape], test_base=request.node.name)


# TODO: these three nullary tensor creation ops can probably be combined in some way
@pytest.mark.parametrize("shapes", [[(128,128)]], ids=["128x128"])
def test_zeros(shapes: List[Shape], request):
    def zeros(builder: TTIRBuilder):
        return builder.zeros(shapes)
    compile_to_flatbuffer(zeros, inputs_shapes=[], test_base=request.node.name)


@pytest.mark.parametrize("shape", [(128,128)], ids=["128x128"])
def test_ones(shape: Shape, request):
    def ones(builder: TTIRBuilder):
        return builder.ones(shape)
    compile_to_flatbuffer(ones, inputs_shapes=[], test_base=request.node.name)


@pytest.mark.parametrize("shape", [(128,128)], ids=["128x128"])
def test_empty(shape: Shape, request):
    def empty(builder: TTIRBuilder):
        return builder.empty(shape)
    compile_to_flatbuffer(empty, inputs_shapes=[], test_base=request.node.name)

@pytest.mark.parametrize("shapes", [[(128,128)]])
@pytest.mark.parametrize("dim", [0,1])
def test_argmax(shapes, dim, request):
    def argmax(in0: Operand, builder: TTIRBuilder):
        return builder.argmax(in0, [dim])
    compile_to_flatbuffer(argmax, inputs_shapes=shapes, test_base=request.node.name)


@pytest.mark.skip("`reverse` doesn't have a legalization. See issue #2495")
@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("dims", [[0, 1]])
def test_reverse(
        shape: Shape,
        dims: List[int],
        request
        ):
    def reverse(in0: Operand, builder: TTIRBuilder):
        return builder.reverse(in0, dims=dims)
    compile_to_flatbuffer(reverse, [shape], test_base=request.node.name)


@pytest.mark.skip("Generated flatbuffer will currently fail to run due to only floats being supported by the runtime. See issue #1775")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_and(
        shape: Shape,
        dim_args: List[int],
        request
        ):
    def reduce_and(in0: Operand, builder: TTIRBuilder):
        return builder.reduce_and(in0, dim_args=dim_args)
    compile_to_flatbuffer(reduce_and, [shape], [torch.bool], test_base=request.node.name)


@pytest.mark.skip("Generated flatbuffer will currently fail to run due to only floats being supported by the runtime. See issue #1775")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_or(
        shape: Shape,
        dim_args: List[int],
        request
        ):
    def reduce_or(in0: Operand, builder: TTIRBuilder):
        return builder.reduce_or(in0, dim_args=dim_args)
    compile_to_flatbuffer(reduce_or, [shape], [torch.bool], test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(2, 3, 4), (3, 4, 2)]])
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
def test_permute(
        shapes: List[Shape],
        permutation: List[int],
        request
        ):
    def permute(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.permute(in0, in1, permutation=DenseI64ArrayAttr.get(permutation))
    compile_to_flatbuffer(permute, shapes, test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(10, 64, 32, 3), (10, 128, 128, 3)]])
@pytest.mark.parametrize("scale_factor", [[2,4]])
def test_upsample2d(shapes: List[Shape], scale_factor: List[int], request):
    def upsample2d(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.upsample2d(in0, in1, scale_factor=DenseI32ArrayAttr.get(scale_factor))
    compile_to_flatbuffer(upsample2d, shapes, test_base=request.node.name)


@pytest.mark.parametrize("shape,start,end,step,dim", [((5,), 0, 5, 1, 0)])
def test_arange(shape: Shape, start: int, end: int, step: int, dim: int, request):
    def arange(in0: Operand, builder: TTIRBuilder):
        return builder.arange(in0, start, end, step, dim)
    compile_to_flatbuffer(arange, [shape], test_base=request.node.name)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("from_type,to_type", [(torch.uint32, torch.uint16)])
def test_typecast(
        shape: Shape,
        from_type: torch.dtype,
        to_type: torch.dtype,
        request):
    def typecast(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.typecast(in0, in1)

    compile_to_flatbuffer(typecast, [shape, shape], [from_type, to_type], test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(4, 4, 128, 128), (4, 4, 128, 128)]])
@pytest.mark.parametrize("dim", [1])
def test_cumsum(shapes: List[Shape], dim: int, request):
    def cumsum(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.cumsum(in0, in1, dim=dim)
    compile_to_flatbuffer(cumsum, shapes, test_base=request.node.name)

def prod(in0: Operand, builder: TTIRBuilder):
    return builder.prod(in0, [1], False)


@pytest.mark.parametrize("shapes", [[(1, 32, 64, 512), (1, 32, 3, 512)]])
def test_fill_cache(shapes: List[Shape], request):
    def fill_cache(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.fill_cache(in0, in1)
    compile_to_flatbuffer(fill_cache, shapes, test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[512, 1024]])
def test_softmax(shapes: List[Shape], request):
    def softmax(in0: Operand, builder: TTIRBuilder):
        return builder.softmax(in0, dimension=-1)
    compile_to_flatbuffer(softmax, shapes, test_base=request.node.name)


@pytest.mark.parametrize("shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]])
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16, torch.int32]])
def test_update_cache(shapes: List[Shape], dtypes: List[torch.dtype], request):
    def update_cache(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        return builder.update_cache(in0, in1, in2)
    compile_to_flatbuffer(update_cache, shapes, inputs_types=dtypes, test_base=request.node.name)

def embedding(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.embedding(in0, in1, in2)


def hoisted_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    # Use op_proxy directly since it accepts ttir_kwargs
    return builder.op_proxy(
        torch.add,
        ttir.AddOp,
        [in0, in1],
        unit_attrs={"should_hoist": UnitAttr.get(builder._ctx)},
        use_zeros=True,
    )


# TODO: look through these thorougly to find ops that might have attributes, break out into their own tests
unary_ops = [exp, expm1, floor, abs, logical_not, neg, sign, cos, sin, tan, atan,
             tanh, log,log1p, relu, gelu, leaky_relu, sqrt, cbrt, rsqrt,
             sigmoid, reciprocal, is_finite, ceil, sum, mean, max, min, prod, get_dimension_size]

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.qint32])
def test_quantize(shape: Shape, scale: float, zero_point: int, dtype: torch.dtype, request):
    def quantize(in0: Operand, builder: TTIRBuilder):
        return builder.quantize(in0, scale, zero_point, dtype)
    compile_to_flatbuffer(quantize, [shape], test_base=request.node.name)


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("input_dtype", [TypeInfo(torch.qint32, 0.1, 0)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_dequantize(shape: Shape, input_dtype: TypeInfo, scale: float, zero_point: int, dtype: torch.dtype, request):
    def dequantize(in0: Operand, builder: TTIRBuilder):
        return builder.dequantize(in0, scale, zero_point, dtype)
    compile_to_flatbuffer(dequantize, [shape], inputs_types=[input_dtype], test_base=request.node.name)



@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("input_dtype", [TypeInfo(torch.qint32, 0.1, 0)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_requantize(shape: Shape, input_dtype: TypeInfo, scale: float, zero_point: int, dtype: torch.dtype, request):
    def requantize(in0: Operand, builder: TTIRBuilder):
        return builder.requantize(in0, scale, zero_point, dtype)
    compile_to_flatbuffer(requantize, [shape], inputs_types=[input_dtype], test_base=request.node.name)


#def test_provided_graph_input_output():
#    def golden_tensor_to_torch_tensor(golden):
#        shape = golden.shape
#        stride = golden.strides
#        match golden.dtype:
#            case DataType.Float16:
#                np_dtype = np.float16
#            case DataType.BFloat16:
#                np_dtype = np.bfloat16
#            case DataType.Float32:
#                np_dtype = np.float32
#            case DataType.Int32:
#                np_dtype = np.int32
#            case None:
#                np_dtype = np.float32
#        np_array = (
#            np.frombuffer(bytes(golden.data), dtype=np_dtype).copy().reshape(shape)
#        )
#        tensor = torch.as_strided(torch.from_numpy(np_array), size=shape, stride=stride)
#        return tensor
#
#    @compile_to_flatbuffer(
#        [
#            (64, 128),
#            (64, 128),
#        ],
#        targets=["ttnn"],
#    )
#    def test_simple_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
#        input_0 = torch.randn(builder.get_shape(in0))
#        input_1 = torch.randn(builder.get_shape(in1))
#        output = input_0 + input_1
#        builder.set_graph_input_output([input_0, input_1], [output])
#        result = builder.add(in0, in1)
#
#        # Verify graph input / output on golden map
#        golden_map = builder.get_golden_map()
#
#        assert "input_0" in golden_map
#        golden_input_0 = golden_tensor_to_torch_tensor(golden_map["input_0"])
#        assert torch.equal(golden_input_0, input_0)
#
#        assert "input_1" in golden_map
#        golden_input_1 = golden_tensor_to_torch_tensor(golden_map["input_1"])
#        assert torch.equal(golden_input_1, input_1)
#
#        assert "output_0" in golden_map
#        golden_output_0 = golden_tensor_to_torch_tensor(golden_map["output_0"])
#        assert torch.equal(golden_output_0, output)
#
#        return result
#
#    test_simple_add()
#
#
#if __name__ == "__main__":
#    import argparse, os

@pytest.mark.parametrize("shape", [(128,128)], ids=str)
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops(
    test_fn: Callable,
    shape: Shape,
    request,
    dtype: torch.dtype = torch.float32):
    compile_to_flatbuffer(test_fn, inputs_shapes=[shape], inputs_types=[dtype], test_base=request.node.name)


@pytest.mark.parametrize("shape", [(128,128)], ids=str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("test_fn", [add, multiply, subtract, eq, ne, le, lt,
                                     ge, gt, div, remainder, maximum, minimum,
                                     pow, matmul, hoisted_add])
def test_binary_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    request):
    """ NOTE: this function is _only_ for binary ops that take the same shape arguments
    """
    compile_to_flatbuffer(test_fn, inputs_shapes=[shape] * 2, inputs_types=[dtype] * 2, test_base=request.node.name)


@pytest.mark.parametrize("shape", [(128,128)], ids=str)
@pytest.mark.parametrize("test_fn", [bitwise_and, bitwise_or, bitwise_xor])
def test_bitwise_binary_ops(
    test_fn: Callable,
    shape: Shape,
    request):
    compile_to_flatbuffer(test_fn, inputs_shapes=[shape] * 2, inputs_types=[torch.int8] * 2, test_base=request.node.name)


@pytest.mark.parametrize("test_fn,inputs_shapes,inputs_dtypes", [
    (transpose,[(64,32)],None),
    (reshape,[(64,32)],None),
    (embedding,[(32, 32), (512, 128)], [torch.bfloat16] * 2),
    pytest.param(where, [(64,64)] * 3, [torch.int8, torch.float32, torch.float32], marks=pytest.mark.skip("Bools aren't supported"))
    ])
def test_unique_ops(
        test_fn: Callable,
        inputs_shapes: List[Shape],
        inputs_dtypes: List[torch.dtype],
        request
        ):
    compile_to_flatbuffer(test_fn, inputs_shapes=inputs_shapes, inputs_types=inputs_dtypes, test_base=request.node.name)
