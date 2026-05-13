# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import *
from ttmlir.dialects import d2m, arith

from ._src.utils import _asindex
from ._src.ast import D2MCompiler, syntax
from ._src.config import config
from ._src.tensor_layout import Layout, float32, float16, bfloat16
from ._src.builder import (
    CompiledKernel,
    LazyTensor,
    kernel,
    to_layout,
    tilize,
    untilize,
    empty,
    view_layout,
    view,
    permute,
    to_host,
)


@syntax("!tensor")
class TensorBlock:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.addf(ast_self, rhs)

    def __sub__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.subf(ast_self, rhs)

    def __mul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.mulf(ast_self, rhs)

    def __neg__(ast_self: TensorBlock) -> TensorBlock:
        return arith.negf(ast_self)

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.divf(ast_self, rhs)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        lhs = ast_self
        assert isinstance(lhs.type, RankedTensorType)
        out_shape = lhs.type.shape
        out_shape[-1] = rhs.type.shape[-1]
        out = d2m.empty(RankedTensorType.get(out_shape, lhs.type.element_type))
        d2m.tile_matmul_block(lhs, rhs, out)
        return out

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return d2m.store(ast_self, rhs)


@syntax("remote_load")
def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
) -> MemTx:
    if mcast_dims is not None:
        if isinstance(mcast_dims, tuple):
            mcast_dims = list(mcast_dims)
        if not isinstance(mcast_dims, list):
            if isinstance(mcast_dims, int):
                mcast_dims = arith.constant(IndexType.get(src.context), mcast_dims)
            mcast_dims = [mcast_dims]
    dst_type = RankedTensorType.get(
        src.type.shape[len(indices) :], src.type.element_type
    )
    dst = d2m.empty(dst_type)
    return d2m.remote_load(
        dst_type,
        src,
        indices,
        mcast_start_index=mcast_start_index,
        mcast_shape=mcast_shape,
        mcast_dims=mcast_dims,
        local_buffer=dst,
    )


@syntax("remote_store")
def remote_store(dst, indices, src):
    return d2m.remote_store(
        dst.type,
        dst,
        indices,
        start_device=[],
        device_mcast_shape=[],
        semaphore_indices=[],
        local_buffer=src,
    )


@syntax(
    "core_index",
    args_as_attr=[
        lambda node: IntegerAttr.get(IntegerType.get_signless(64), node.value)
    ],
)
def core_index(index):
    return d2m.core_index(index)


# --- Tile Unary Compute Ops ---


@syntax("tile_recip")
def tile_recip(input):
    return d2m.tile_recip(input.type, input)


@syntax("tile_exp")
def tile_exp(input):
    return d2m.tile_exp(input.type, input)


@syntax("tile_log")
def tile_log(input):
    return d2m.tile_log(input.type, input)


@syntax("tile_negative")
def tile_negative(input):
    return d2m.tile_negative(input.type, input)


@syntax("tile_cos")
def tile_cos(input):
    return d2m.tile_cos(input.type, input)


@syntax("tile_acos")
def tile_acos(input):
    return d2m.tile_acos(input.type, input)


@syntax("tile_sin")
def tile_sin(input):
    return d2m.tile_sin(input.type, input)


@syntax("tile_asin")
def tile_asin(input):
    return d2m.tile_asin(input.type, input)


@syntax("tile_tan")
def tile_tan(input):
    return d2m.tile_tan(input.type, input)


@syntax("tile_atan")
def tile_atan(input):
    return d2m.tile_atan(input.type, input)


@syntax("tile_tanh")
def tile_tanh(input):
    return d2m.tile_tanh(input.type, input)


@syntax("tile_sqrt")
def tile_sqrt(input):
    return d2m.tile_sqrt(input.type, input)


@syntax("tile_rsqrt")
def tile_rsqrt(input):
    return d2m.tile_rsqrt(input.type, input)


@syntax("tile_sigmoid")
def tile_sigmoid(input):
    return d2m.tile_sigmoid(input.type, input)


@syntax("tile_hardsigmoid")
def tile_hardsigmoid(input):
    return d2m.tile_hardsigmoid(input.type, input)


@syntax("tile_silu")
def tile_silu(input):
    return d2m.tile_silu(input.type, input)


@syntax("tile_relu")
def tile_relu(input):
    return d2m.tile_relu(input.type, input)


@syntax("tile_gelu")
def tile_gelu(input):
    return d2m.tile_gelu(input.type, input)


@syntax("tile_erf")
def tile_erf(input):
    return d2m.tile_erf(input.type, input)


@syntax("tile_erfc")
def tile_erfc(input):
    return d2m.tile_erfc(input.type, input)


@syntax("tile_sign")
def tile_sign(input):
    return d2m.tile_sign(input.type, input)


@syntax("tile_ceil")
def tile_ceil(input):
    return d2m.tile_ceil(input.type, input)


@syntax("tile_floor")
def tile_floor(input):
    return d2m.tile_floor(input.type, input)


@syntax("tile_abs")
def tile_abs(input):
    return d2m.tile_abs(input.type, input)


@syntax("tile_bitwise_not")
def tile_bitwise_not(input):
    return d2m.tile_bitwise_not(input.type, input)


@syntax("tile_logical_not")
def tile_logical_not(input):
    return d2m.tile_logical_not(input.type, input)


@syntax("tile_eqz")
def tile_eqz(input):
    return d2m.tile_eqz(input.type, input)


@syntax("tile_nez")
def tile_nez(input):
    return d2m.tile_nez(input.type, input)


@syntax("tile_gtz")
def tile_gtz(input):
    return d2m.tile_gtz(input.type, input)


@syntax("tile_gez")
def tile_gez(input):
    return d2m.tile_gez(input.type, input)


@syntax("tile_ltz")
def tile_ltz(input):
    return d2m.tile_ltz(input.type, input)


@syntax("tile_lez")
def tile_lez(input):
    return d2m.tile_lez(input.type, input)


@syntax("tile_typecast")
def tile_typecast(input, result_type):
    return d2m.tile_typecast(result_type, input)


@syntax("tile_transpose")
def tile_transpose(input):
    return d2m.tile_transpose(input.type, input)


# --- Tile Binary Compute Ops ---


@syntax("tile_add")
def tile_add(lhs, rhs):
    return d2m.tile_add(lhs.type, lhs, rhs)


@syntax("tile_sub")
def tile_sub(lhs, rhs):
    return d2m.tile_sub(lhs.type, lhs, rhs)


@syntax("tile_mul")
def tile_mul(lhs, rhs):
    return d2m.tile_mul(lhs.type, lhs, rhs)


@syntax("tile_div")
def tile_div(lhs, rhs):
    return d2m.tile_div(lhs.type, lhs, rhs)


@syntax("tile_pow")
def tile_pow(lhs, rhs):
    return d2m.tile_pow(lhs.type, lhs, rhs)


@syntax("tile_maximum")
def tile_maximum(lhs, rhs):
    return d2m.tile_maximum(lhs.type, lhs, rhs)


@syntax("tile_minimum")
def tile_minimum(lhs, rhs):
    return d2m.tile_minimum(lhs.type, lhs, rhs)


@syntax("tile_bitwise_and")
def tile_bitwise_and(lhs, rhs):
    return d2m.tile_bitwise_and(lhs.type, lhs, rhs)


@syntax("tile_bitwise_or")
def tile_bitwise_or(lhs, rhs):
    return d2m.tile_bitwise_or(lhs.type, lhs, rhs)


@syntax("tile_bitwise_xor")
def tile_bitwise_xor(lhs, rhs):
    return d2m.tile_bitwise_xor(lhs.type, lhs, rhs)


# --- Tile Ternary / Special Compute Ops ---


@syntax("tile_where")
def tile_where(condition, true_value, false_value):
    return d2m.tile_where(true_value.type, condition, true_value, false_value)


@syntax("tile_matmul")
def tile_matmul(a, b, c):
    return d2m.tile_matmul(c.type, a, b, c)


@syntax("tile_clamp_scalar")
def tile_clamp_scalar(input, min, max):
    return d2m.tile_clamp_scalar(input.type, input, min, max)


@syntax("tile_reduce_sum")
def tile_reduce_sum(a, b, c, reduce_dim):
    return d2m.tile_reduce_sum(c.type, a, b, c, reduce_dim)


@syntax("tile_reduce_max")
def tile_reduce_max(a, b, c, reduce_dim):
    return d2m.tile_reduce_max(c.type, a, b, c, reduce_dim)


@syntax("tile_reduce_mean")
def tile_reduce_mean(a, b, c, reduce_dim):
    return d2m.tile_reduce_mean(c.type, a, b, c, reduce_dim)


@syntax("tile_bcast")
def tile_bcast(input, bcast_type):
    return d2m.tile_bcast(input.type, input, bcast_type)


@syntax("fill_tile")
def fill_tile(value, result_type):
    return d2m.fill_tile(result_type, value)


@syntax("tile_tilize_block")
def tile_tilize_block(input, output):
    return d2m.tile_tilize_block(output.type, input, output)


@syntax("tile_untilize_block")
def tile_untilize_block(input, output):
    return d2m.tile_untilize_block(output.type, input, output)


@syntax("!d2m.semaphore")
class Semaphore:
    def set(ast_self, value, core=None, mcast=None):
        return d2m.semaphore_set(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def inc(ast_self, value, core=None, mcast=None):
        return d2m.semaphore_inc(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def wait(ast_self, value, reset=None):
        return d2m.semaphore_wait(
            ast_self, _asindex(value), reset_value=_asindex(reset)
        )


