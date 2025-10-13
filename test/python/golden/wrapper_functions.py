# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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
from builder.base.builder_golden import BuilderGoldenTensor
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)


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


def atan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.atan(in0, unit_attrs=unit_attrs)


def tan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    import math

    builder_input_golden = builder._get_golden_tensor(in0)
    randn_tensor = torch.randn(
        builder_input_golden.shape, dtype=builder_input_golden.dtype
    )
    input_golden = randn_tensor.uniform_((-math.pi / 2 + 0.05), (math.pi / 2 - 0.05))
    output_golden = torch.tan(input_golden)
    tan_0 = builder.tan(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
    return tan_0


def tanh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)


def logical_not(
    in0: Operand,
    builder: TTIRBuilder,
    shape: Shape,
    dtype: torch.dtype,
    unit_attrs: Optional[List[str]] = None,
):
    builder_input_golden = builder._get_golden_tensor(in0)
    dtype = builder_input_golden.dtype
    randn_tensor = torch.randn(builder_input_golden.shape, dtype=dtype)
    input_tensor = randn_tensor.uniform_(-10.0, 10.0)
    input_tensor[torch.abs(input_tensor) < 4.0] = 0.0
    input_tensor = input_tensor.to(dtype)
    # Torch returns bool tensor but ttnn doesn't have bool type, convert to input dtype.
    golden_output_tensor = torch.logical_not(input_tensor).to(dtype)
    logical_not_0 = builder.logical_not(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_tensor}, {logical_not_0: golden_output_tensor})
    return logical_not_0


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


def log(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    builder_input_golden = builder._get_golden_tensor(in0)
    randn_tensor = torch.randn(
        builder_input_golden.shape, dtype=builder_input_golden.dtype
    )
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, 0.01)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log(input_golden)
    log_0 = builder.log(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_golden}, {log_0: output_golden})
    return log_0


def log1p(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    builder_input_golden = builder._get_golden_tensor(in0)
    randn_tensor = torch.randn(
        builder_input_golden.shape, dtype=builder_input_golden.dtype
    )
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, -0.99)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log1p(input_golden)
    log1p_0 = builder.log1p(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_golden}, {log1p_0: output_golden})
    return log1p_0


def relu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)


def gelu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.gelu(in0, unit_attrs=unit_attrs)


def clamp_scalar(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.clamp_scalar(
        in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
    )


def clamp_tensor(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)


def sqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    builder_input_golden = builder._get_golden_tensor(in0)
    randn_tensor = torch.randn(
        builder_input_golden.shape, dtype=builder_input_golden.dtype
    )
    golden_output_tensor = torch.sqrt(input_tensor)
    sqrt_0 = builder.sqrt(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_tensor}, {sqrt_0: golden_output_tensor})
    return sqrt_0


def leaky_relu(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.leaky_relu(in0, unit_attrs=unit_attrs)


def cbrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cbrt(in0, unit_attrs=unit_attrs)


def rsqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    builder_input_golden = builder._get_golden_tensor(in0)
    randn_tensor = torch.randn(
        builder_input_golden.shape, dtype=builder_input_golden.dtype
    )
    golden_output_tensor = torch.rsqrt(input_tensor)
    rsqrt_0 = builder.rsqrt(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_tensor}, {rsqrt_0: golden_output_tensor})
    return rsqrt_0


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
    dividend_tensor = builder._get_golden_tensor(in0)
    divisor_tensor = builder._get_golden_tensor(in1)

    dividend_tensor = dividend_tensor.apply_shardwise(
        lambda shard: (
            shard.__setitem__(shard.abs() < 0.01, 0.03) or shard
            if torch.is_floating_point(shard)
            else shard
        )
    )

    divisor_tensor = divisor_tensor.apply_shardwise(
        lambda shard: (
            shard.__setitem__(shard.abs() < 0.01, -0.03) or shard
            if torch.is_floating_point(shard)
            else shard
        )
    )

    output_golden = torch.div(dividend_tensor, divisor_tensor)
    div0 = builder.div(in0, in1, unit_attrs=unit_attrs)
    builder.set_goldens_from_builder_tensor(
        {in0: dividend_tensor, in1: divisor_tensor}, {div0: output_golden}
    )
    return div0


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


def linear(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.linear(in0, in1, in2, unit_attrs=unit_attrs)


def pow(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    randn_base_tensor = builder._get_golden_tensor(in0)
    randn_exponent_tensor = builder._get_golden_tensor(in1)

    randn_base_tensor = randn_base_tensor.apply_shardwise(
        lambda shard: (
            shard.abs() if torch.is_floating_point(randn_exponent_tensor) else shard
        )
    )

    if torch.is_floating_point(randn_exponent_tensor):
        randn_base_tensor = torch.abs(randn_base_tensor)
    output_golden = torch.pow(randn_base_tensor, randn_exponent_tensor)
    pow0 = builder.pow(in0, in1, unit_attrs=unit_attrs)
    builder.set_goldens_from_builder_tensor(
        {in0: randn_base_tensor, in1: randn_exponent_tensor}, {pow0: output_golden}
    )
    return pow0


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


def squeeze(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.squeeze(in0, unit_attrs=unit_attrs)


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
    builder: TTIRBuilder,
    broadcast_dimensions: Optional[List[int]] = None,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.broadcast(
        in0, broadcast_dimensions=broadcast_dimensions, unit_attrs=unit_attrs
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


def permute(
    in0: Operand,
    permutation: List[int],
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.permute(
        in0,
        permutation=permutation,
        unit_attrs=unit_attrs,
    )


funcs_with_other_args = {
    permute: {"shapes": [(2, 3, 5)], "args": {"permutation": [2, 0, 1]}},
}
