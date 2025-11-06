# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional
from conftest import x86_only
from builder.base.builder import Operand, Shape, Builder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder

# from builder.d2m.d2m_builder import D2MBuilder
from builder.base import get_golden_function
from builder.base.builder_utils import (
    compile_and_execute_ttnn,
)
from test_utils import (
    Marks,
    shape_str,
)
from ttmlir.dialects import ttnn


# Unary Ops
def abs(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.abs(in0, unit_attrs=unit_attrs)


def atan(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.atan(in0, unit_attrs=unit_attrs)


def cbrt(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.cbrt(in0, unit_attrs=unit_attrs)


def ceil(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.ceil(in0, unit_attrs=unit_attrs)


def erf(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.erf(in0, unit_attrs=unit_attrs)


def erfc(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.erfc(in0, unit_attrs=unit_attrs)


def exp(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.exp(in0, unit_attrs=unit_attrs)


def expm1(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.expm1(in0, unit_attrs=unit_attrs)


def floor(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.floor(in0, unit_attrs=unit_attrs)


def cos(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.cos(in0, unit_attrs=unit_attrs)


def gelu(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.gelu(in0, unit_attrs=unit_attrs)


def isfinite(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.isfinite(in0, unit_attrs=unit_attrs)


# Special handling for log PCC checks. Due to the vertical asymptote on the log graph,
# small changes in input values result in large changes in output values at negative values,
# so both graph and golden tensors must be constrained accordingly.
def log(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    log_0 = builder.log(in0, unit_attrs=unit_attrs)

    # Constrain values for log
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("log op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, 0.01)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log(input_golden)
    builder.set_goldens({in0: input_golden}, {log_0: output_golden})
    return log_0


def log1p(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    log1p_0 = builder.log1p(in0, unit_attrs=unit_attrs)

    # Constrain values for log1p
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("log1p op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, -0.99)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log1p(input_golden)

    builder.set_goldens({in0: input_golden}, {log1p_0: output_golden})
    return log1p_0


def logical_not(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.logical_not(in0, unit_attrs=unit_attrs)


def neg(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)


def relu(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)


def reciprocal(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    reciprocal_0 = builder.reciprocal(in0, unit_attrs=unit_attrs)

    # Constrain values for reciprocal
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("reciprocal op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    input = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    input_safe = torch.clamp(input, min=-1e-6, max=None)
    input_safe = torch.where(input_safe == 0, torch.tensor(1e-6), input_safe)
    golden_output = torch.reciprocal(input_safe)
    builder.set_goldens({in0: input_safe}, {reciprocal_0: golden_output})
    return reciprocal_0


def relu6(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.relu6(in0, unit_attrs=unit_attrs)


def rsqrt(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    rsqrt_0 = builder.rsqrt(in0, unit_attrs=unit_attrs)
    # Constrain values for rsqrt
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("rsqrt op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    input_tensor = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    golden_output_tensor = torch.rsqrt(input_tensor)
    builder.set_goldens({in0: input_tensor}, {rsqrt_0: golden_output_tensor})
    return rsqrt_0


def sigmoid(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)


def sign(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.sign(in0, unit_attrs=unit_attrs)


def silu(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.silu(in0, unit_attrs=unit_attrs)


def sqrt(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    sqrt_0 = builder.sqrt(in0, unit_attrs=unit_attrs)

    # Constrain values for sqrt
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("rsqrt op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    input_tensor = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    golden_output_tensor = torch.sqrt(input_tensor)
    builder.set_goldens({in0: input_tensor}, {sqrt_0: golden_output_tensor})
    return sqrt_0


def sin(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)


# Special handling for log PCC checks. Due to the vertical asymptote on the log graph,
# small changes in input values result in large changes in output values at negative values,
# so both graph and golden tensors must be constrained accordingly.
def tan(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    tan_0 = builder.tan(in0, unit_attrs=unit_attrs)

    # Constrain values for tan
    import math

    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("tan op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    input_golden = randn_tensor.uniform_((-math.pi / 2 + 0.05), (math.pi / 2 - 0.05))
    output_golden = torch.tan(input_golden)
    builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
    return tan_0


def tanh(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)


# TTNNBuilder unary ops
def mish(
    in0: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.mish(in0, unit_attrs=unit_attrs)


# StableHLOBuilder unary ops
def cosine(
    in0: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.cosine(in0, unit_attrs=unit_attrs)


def sine(
    in0: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.sine(in0, unit_attrs=unit_attrs)


def logistic(
    in0: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.logistic(in0, unit_attrs=unit_attrs)


# Bitwise unary ops (int only)
def bitwise_not(in0: Operand, builder: Builder, unit_attrs: Optional[List[str]] = None):
    return builder.bitwise_not(in0, unit_attrs=unit_attrs)


# Unary ops with float parameter
def leaky_relu(
    in0: Operand,
    parameter: float,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.leaky_relu(in0, parameter, unit_attrs=unit_attrs)


# Binary ops
def add(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def atan2(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.atan2(in0, in1, unit_attrs=unit_attrs)


def divide(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.divide(in0, in1, unit_attrs=unit_attrs)


def logical_and(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def logical_or(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_or(in0, in1, unit_attrs=unit_attrs)


def logical_xor(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_xor(in0, in1, unit_attrs=unit_attrs)


def maximum(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.maximum(in0, in1, unit_attrs=unit_attrs)


def minimum(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.minimum(in0, in1, unit_attrs=unit_attrs)


def multiply(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def remainder(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


# Binary comparison ops
def eq(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.eq(in0, in1, unit_attrs=unit_attrs)


def ge(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ge(in0, in1, unit_attrs=unit_attrs)


def gt(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gt(in0, in1, unit_attrs=unit_attrs)


def le(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.le(in0, in1, unit_attrs=unit_attrs)


def lt(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.lt(in0, in1, unit_attrs=unit_attrs)


def ne(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ne(in0, in1, unit_attrs=unit_attrs)


# TTIRBuilder-specific binary ops
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


# TTNNBuilder binary ops
def pow_tensor(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.pow_tensor(in0, in1, unit_attrs=unit_attrs)


# TTNNBuilder Binary logical shift ops (int only)
def logical_left_shift(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    logical_left_shift_0 = builder.logical_left_shift(in0, in1, unit_attrs=unit_attrs)
    # Constrain shift amounts to be within valid range
    shift_tensor_1 = builder._get_golden_tensor(in1)
    dtype_bits = torch.iinfo(shift_tensor_1.shard_at(0).dtype).bits
    # Handle uint32 which doesn't support % operator in PyTorch
    constrained_shift_tensor = shift_tensor_1.apply_shardwise(
        lambda shard: (shard.to(torch.int64) % dtype_bits).to(shard.dtype)
    )

    golden_fn = get_golden_function(ttnn.LogicalLeftShiftOp)
    output_golden = golden_fn(builder._get_golden_tensor(in0), constrained_shift_tensor)
    builder.set_goldens_from_builder_tensor(
        {in1: constrained_shift_tensor}, {logical_left_shift_0: output_golden}
    )
    return logical_left_shift_0


def logical_right_shift(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    logical_right_shift_0 = builder.logical_right_shift(in0, in1, unit_attrs=unit_attrs)
    # Constrain shift amounts to be within valid range
    shift_tensor_1 = builder._get_golden_tensor(in1)
    dtype_bits = torch.iinfo(shift_tensor_1.shard_at(0).dtype).bits
    # Handle uint32 which doesn't support % operator in PyTorch
    constrained_shift_tensor = shift_tensor_1.apply_shardwise(
        lambda shard: (shard.to(torch.int64) % dtype_bits).to(shard.dtype)
    )

    golden_fn = get_golden_function(ttnn.LogicalRightShiftOp)
    output_golden = golden_fn(builder._get_golden_tensor(in0), constrained_shift_tensor)
    builder.set_goldens_from_builder_tensor(
        {in1: constrained_shift_tensor}, {logical_right_shift_0: output_golden}
    )
    return logical_right_shift_0


# Binary bitwise ops (int only)
def bitwise_and(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_and(in0, in1, unit_attrs=unit_attrs)


def bitwise_or(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_or(in0, in1, unit_attrs=unit_attrs)


def bitwise_xor(
    in0: Operand,
    in1: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_xor(in0, in1, unit_attrs=unit_attrs)


binary_bitwise_ops = [
    bitwise_and,
    bitwise_or,
    bitwise_xor,
]


binary_bitwise_dtypes = [
    torch.int32,
    torch.uint32,
    torch.uint16,
    torch.uint8,
]


# Ternary Ops
def where(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: Builder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.where(in0, in1, in2, unit_attrs=unit_attrs)


ternary_ops = [
    where | Marks(pytest.mark.xfail(reason="Fails Golden")),
]
