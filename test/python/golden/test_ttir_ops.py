# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect
import torch
import numpy as np

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder, Attribute, UnitAttr, TypeInfo
from ttmlir.dialects import ttir
from ttmlir.ir import *
from ttmlir.passes import GoldenTensor, DataType


# NOTE: This test is not valid for TTRT Perf due to weird issues with perf collection. Issue #2371
"""
@compile_to_flatbuffer([(1, 128, 128, 1)], targets=["ttnn"])
def test_squeeze(in0: Operand, builder: TTIRBuilder):
    return builder.squeeze(in0, 0)
"""


# NOTE: Same as Squeeze, this Op is not valid for TTRT Perf. Issue #2371
"""
@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_unsqueeze(in0: Operand, builder: TTIRBuilder):
    return builder.unsqueeze(in0, 0)
"""


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_exp(in0: Operand, builder: TTIRBuilder):
    return builder.exp(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_expm1(in0: Operand, builder: TTIRBuilder):
    return builder.expm1(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_ceil(in0: Operand, builder: TTIRBuilder):
    return builder.ceil(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_floor(in0: Operand, builder: TTIRBuilder):
    return builder.floor(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_abs(in0: Operand, builder: TTIRBuilder):
    return builder.abs(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_logical_not(in0: Operand, builder: TTIRBuilder):
    return builder.logical_not(in0)


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
"""
@compile_to_flatbuffer([(128, 128)], inputs_types=[torch.int8], targets=["ttnn"])
def test_bitwise_not(in0: Operand, builder: TTIRBuilder):
    return builder.bitwise_not(in0)
"""


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_neg(in0: Operand, builder: TTIRBuilder):
    return builder.neg(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sign(in0: Operand, builder: TTIRBuilder):
    return builder.sign(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sin(in0: Operand, builder: TTIRBuilder):
    return builder.sin(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_cos(in0: Operand, builder: TTIRBuilder):
    return builder.cos(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_tan(in0: Operand, builder: TTIRBuilder):
    return builder.tan(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_atan(in0: Operand, builder: TTIRBuilder):
    return builder.atan(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_tanh(in0: Operand, builder: TTIRBuilder):
    return builder.tanh(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_log(in0: Operand, builder: TTIRBuilder):
    return builder.log(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_log1p(in0: Operand, builder: TTIRBuilder):
    return builder.log1p(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_relu(in0: Operand, builder: TTIRBuilder):
    return builder.relu(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_gelu(in0: Operand, builder: TTIRBuilder):
    return builder.gelu(in0)


@compile_to_flatbuffer([(64, 128)], inputs_types=[torch.bfloat16], targets=["ttnn"])
def test_clamp_scalar(in0: Operand, builder: TTIRBuilder):
    return builder.clamp_scalar(in0, max_arg=3.0, min_arg=2.0)


@compile_to_flatbuffer([(32, 64), (32, 64), (32, 64), (32, 64)], targets=["ttnn"])
def test_clamp_tensor(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand, builder: TTIRBuilder
):
    return builder.clamp_tensor(in0, in1, in2, in3)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_leaky_relu(in0: Operand, builder: TTIRBuilder):
    return builder.leaky_relu(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sqrt(in0: Operand, builder: TTIRBuilder):
    return builder.sqrt(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_cbrt(in0: Operand, builder: TTIRBuilder):
    return builder.cbrt(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_rsqrt(in0: Operand, builder: TTIRBuilder):
    return builder.rsqrt(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sigmoid(in0: Operand, builder: TTIRBuilder):
    return builder.sigmoid(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_reciprocal(in0: Operand, builder: TTIRBuilder):
    return builder.reciprocal(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_is_finite(in0: Operand, builder: TTIRBuilder):
    return builder.is_finite(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_get_dimension_size(in0: Operand, builder: TTIRBuilder):
    return builder.get_dimension_size(in0)


@compile_to_flatbuffer(
    [
        (4, 10, 3, 5, 7),
        (4, 10, 5, 7, 3),
    ],
    targets=["ttnn"],
)
def test_dot_general(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.dot_general(in0, in1, [0], [3], [0], [2])


@compile_to_flatbuffer(
    [
        (64, 128),
        (32, 128),
        (16, 128),
    ],
    targets=["ttnn"],
)
def test_concat(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.concat([in0, in1, in2])


@compile_to_flatbuffer(
    [
        (64, 128),
        (64, 128),
    ],
    targets=["ttnn"],
)
def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.multiply(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_logical_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_and(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_logical_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_or(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_logical_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_xor(in0, in1)


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
"""
@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.int8],
    targets=["ttnn"],
)
def test_bitwise_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_and(in0, in1)
"""


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
"""
@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.int8],
    targets=["ttnn"],
)
def test_bitwise_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_or(in0, in1)
"""


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
"""
@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.int8],
    targets=["ttnn"],
)
def test_bitwise_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_xor(in0, in1)
"""


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_subtract(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.subtract(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_eq(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.eq(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_ne(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ne(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_ge(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ge(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_gt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.gt(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_le(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.le(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_lt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.lt(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_div(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.div(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_remainder(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.remainder(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_maximum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.maximum(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_minimum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.minimum(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_pow(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.pow(in0, in1)


@compile_to_flatbuffer(
    [
        (10, 64, 32),
        (32, 128),
        (128,),
    ],
    inputs_types=[torch.bfloat16, torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_linear(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.linear(in0, in1, in2)


@compile_to_flatbuffer(
    [
        (32, 64),
        (64, 128),
    ],
    targets=["ttnn"],
)
def test_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.matmul(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_sum(in0: Operand, builder: TTIRBuilder):
    return builder.sum(in0)


@compile_to_flatbuffer(
    [
        (128, 128),
    ],
    targets=["ttnn"],
)
def test_mean(in0: Operand, builder: TTIRBuilder):
    return builder.mean(in0)


@compile_to_flatbuffer(
    [
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_max(in0: Operand, builder: TTIRBuilder):
    return builder.max(in0)


@compile_to_flatbuffer(
    [
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_min(in0: Operand, builder: TTIRBuilder):
    return builder.min(in0)


@compile_to_flatbuffer(
    [
        (32, 64),
    ],
    targets=["ttnn"],
)
def test_reshape(in0: Operand, builder: TTIRBuilder):
    return builder.reshape(in0, [2048])


@compile_to_flatbuffer(
    [
        (32, 64),
    ],
    targets=["ttnn"],
)
def test_transpose(in0: Operand, builder: TTIRBuilder):
    return builder.transpose(in0)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.float32, torch.float32],
    targets=["ttnn"],
)
def test_where(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.where(in0, in1, in2)


@compile_to_flatbuffer(
    [
        (1, 32, 32),
    ],
    targets=["ttnn"],
)
def test_repeat(in0: Operand, builder: TTIRBuilder):
    return builder.repeat(in0, [32, 1, 1])


@compile_to_flatbuffer(
    [
        (1, 8, 1, 12, 64),
        (1, 8, 1, 12, 64),
    ],
    targets=["ttnn"],
)
def test_repeat_interleave(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.repeat_interleave(in0, in1, repeats=1, dim=0)


@compile_to_flatbuffer(
    [
        (1, 1, 32),
        (1, 16, 32),
    ],
    targets=["ttnn"],
)
def test_broadcast(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.broadcast(in0, in1, [1, 16, 1])


@compile_to_flatbuffer(
    [
        (1, 32, 32, 64),
        (64, 32, 3, 3),
        (1, 1, 1, 64),
        (1, 16, 28, 64),
    ],
    inputs_types=[torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_conv2d(
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
    ],
    inputs_types=[torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_conv_transpose2d(
    in0: Operand, weight: Operand, bias: Operand, in1: Operand, builder: TTIRBuilder
):
    stride = IntegerAttr.get(IntegerType.get_signless(32), 1)
    padding = IntegerAttr.get(IntegerType.get_signless(32), 0)
    output_padding = IntegerAttr.get(IntegerType.get_signless(32), 0)
    dilation = IntegerAttr.get(IntegerType.get_signless(32), 1)
    return builder.conv_transpose2d(
        in0,
        weight,
        bias,
        in1,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=1,
    )


@compile_to_flatbuffer(
    [
        (1, 128, 128, 32),
        (1, 64, 64, 32),
    ],
    inputs_types=[torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_max_pool2d(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.max_pool2d(
        in0,
        in1,
        kernel_height=2,
        kernel_width=2,
        stride_height=2,
        stride_width=2,
        dilation_height=1,
        dilation_width=1,
        ceil_mode=False,
        padding_left=0,
        padding_right=0,
        padding_top=0,
        padding_bottom=0,
    )


@compile_to_flatbuffer(
    [(32, 32), (34, 33)],
    targets=["ttnn"],
)
def test_pad(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.pad(in0, in1, padding=[0, 2, 1, 0], value=0)


@compile_to_flatbuffer([(32, 64)], targets=["ttnn"])
def test_index(in0: Operand, builder: TTIRBuilder):
    return builder.index(in0)


# NOTE: select thowing floating point exception. Issue #2496
# @compile_to_flatbuffer([(4, 4)], targets=["ttnn"])
# def test_select(in0: Operand, builder: TTIRBuilder):
# return builder.select(in0, dim = 1, begin = 2, length = 2)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_zeros(in0: Operand, builder: TTIRBuilder):
    return builder.zeros([128, 128])


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_ones(in0: Operand, builder: TTIRBuilder):
    return builder.ones([128, 128])


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_empty(in0: Operand, builder: TTIRBuilder):
    return builder.empty([128, 128])


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_argmax(in0: Operand, builder: TTIRBuilder):
    return builder.argmax(in0, [1])


# TODO: #Resolve "RuntimeError: Failed to run pass manager. failed to legalize operation 'ttir.reverse'."
# may not be supported by ttir_to_ttnn_backend_pipeline. Issue #2495
# @compile_to_flatbuffer([(64, 64)], targets=["ttnn"])
# def test_reverse(in0: Operand, builder: TTIRBuilder):
#    return builder.reverse(in0, [0,1])


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
# @compile_to_flatbuffer([(4, 4)], inputs_types=[torch.bool], targets=["ttnn"])
# def test_reduce_and(in0: Operand, builder: TTIRBuilder):
# return builder.reduce_and(in0, dim_args=[0,1])


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
# @compile_to_flatbuffer([(128, 128)], inputs_types=[torch.bool], targets=["ttnn"])
# def test_reduce_or(in0: Operand, builder: TTIRBuilder):
# return builder.reduce_or(in0, dim_args=[0,1])


@compile_to_flatbuffer([(2, 3, 4), (3, 4, 2)], targets=["ttnn"])
def test_permute(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.permute(in0, in1, permutation=DenseI64ArrayAttr.get([1, 2, 0]))


@compile_to_flatbuffer(
    [(10, 64, 32, 3), (10, 128, 128, 3)],
    inputs_types=[torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_upsample2d(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.upsample2d(in0, in1, scale_factor=DenseI32ArrayAttr.get([2, 4]))


@compile_to_flatbuffer([(5,)], inputs_types=[torch.bfloat16], targets=["ttnn"])
def test_arange(in0: Operand, builder: TTIRBuilder):
    return builder.arange(in0, 0, 5, 1, 0)


@compile_to_flatbuffer(
    [(32, 32), (32, 32)], inputs_types=[torch.uint32, torch.uint16], targets=["ttnn"]
)
def test_typecast(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.typecast(in0, in1)


@compile_to_flatbuffer(
    [(128, 10, 32, 4)],
    inputs_types=[torch.bfloat16],
    targets=["ttnn"],
)
def test_prod(in0: Operand, builder: TTIRBuilder):
    return builder.prod(in0, [1])


@compile_to_flatbuffer(
    [(4, 4, 128, 128), (4, 4, 128, 128)],
    inputs_types=[torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_cumsum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.cumsum(in0, in1, dim=1)


@compile_to_flatbuffer(
    [(32, 32), (512, 128)],
    inputs_types=[torch.bfloat16, torch.bfloat16],
    targets=["ttnn"],
)
def test_embedding(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.embedding(in0, in1)


@compile_to_flatbuffer(
    [
        (1, 32, 64, 512),
        (1, 32, 3, 512),
    ],
    targets=["ttnn"],
)
def test_fill_cache(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.fill_cache(in0, in1)


@compile_to_flatbuffer(
    [
        (1, 32, 64, 512),
        (1, 32, 1, 512),
        (1,),
    ],
    inputs_types=[torch.bfloat16, torch.bfloat16, torch.int32],
    targets=["ttnn"],
)
def test_update_cache(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.update_cache(in0, in1, in2)


@compile_to_flatbuffer(
    [
        (32, 32),
        (32, 32),
        (32, 32),
    ],
    targets=["ttnn"],
)
def test_arbitrary_op_chain(
    in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
):
    add = builder.add(in0, in1)
    exp = builder.exp(in2)
    return builder.multiply(add, exp)


@compile_to_flatbuffer(
    [
        (64, 128),
        (64, 128),
    ],
    targets=["ttnn"],
)
def test_hoisted_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    # Use op_proxy directly since it accepts ttir_kwargs
    return builder.op_proxy(
        torch.add,
        ttir.AddOp,
        [in0, in1],
        unit_attrs={"should_hoist": UnitAttr.get(builder._ctx)},
        use_zeros=True,
    )


@compile_to_flatbuffer(
    [(128, 128)],
    targets=["ttnn"],
)
def test_quantize(in0: Operand, builder: TTIRBuilder):
    return builder.quantize(in0, scale=0.1, zero_point=0, dtype=torch.qint32)


@compile_to_flatbuffer(
    [(128, 128)],
    targets=["ttnn"],
    inputs_types=[TypeInfo(dtype=torch.qint32, scale=0.1, zero_point=0)],
)
def test_dequantize(in0: Operand, builder: TTIRBuilder):
    return builder.dequantize(in0, scale=0.1, zero_point=0, dtype=torch.float32)


@compile_to_flatbuffer(
    [(128, 128)],
    targets=["ttnn"],
    inputs_types=[TypeInfo(dtype=torch.qint32, scale=0.1, zero_point=0)],
)
def test_requantize(in0: Operand, builder: TTIRBuilder):
    return builder.requantize(in0, scale=0.2, zero_point=0, dtype=torch.qint32)


def test_provided_graph_input_output():
    def golden_tensor_to_torch_tensor(golden):
        shape = golden.shape
        stride = golden.strides
        match golden.dtype:
            case DataType.Float16:
                np_dtype = np.float16
            case DataType.BFloat16:
                np_dtype = np.bfloat16
            case DataType.Float32:
                np_dtype = np.float32
            case DataType.Int32:
                np_dtype = np.int32
            case None:
                np_dtype = np.float32
        np_array = (
            np.frombuffer(bytes(golden.data), dtype=np_dtype).copy().reshape(shape)
        )
        tensor = torch.as_strided(torch.from_numpy(np_array), size=shape, stride=stride)
        return tensor

    @compile_to_flatbuffer(
        [
            (64, 128),
            (64, 128),
        ],
        targets=["ttnn"],
    )
    def test_simple_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        input_0 = torch.randn(builder.get_shape(in0))
        input_1 = torch.randn(builder.get_shape(in1))
        output = input_0 + input_1
        builder.set_graph_input_output([input_0, input_1], [output])
        result = builder.add(in0, in1)

        # Verify graph input / output on golden map
        golden_map = builder.get_golden_map()

        assert "input_0" in golden_map
        golden_input_0 = golden_tensor_to_torch_tensor(golden_map["input_0"])
        assert torch.equal(golden_input_0, input_0)

        assert "input_1" in golden_map
        golden_input_1 = golden_tensor_to_torch_tensor(golden_map["input_1"])
        assert torch.equal(golden_input_1, input_1)

        assert "output_0" in golden_map
        golden_output_0 = golden_tensor_to_torch_tensor(golden_map["output_0"])
        assert torch.equal(golden_output_0, output)

        return result

    test_simple_add()


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run TTIR Builder Op tests")
    parser.add_argument(
        "--path",
        type=str,
        help="Optional output path for the flatbuffer. Creates path if supplied path doesn't exist",
    )
    args = parser.parse_args()

    if args.path and os.path.exists(args.path):
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        set_output_path(args.path)

    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )

    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
