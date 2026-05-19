# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import *
from ttmlir.dialects import d2m, arith, linalg, ttcore

from ._src.utils import _asindex
from ._src.ast import D2MCompiler, syntax
from ._src.config import config
from ._src.errors import D2mJitError
from ._src.tensor_layout import Layout, float32, float16, bfloat16
from ._src.builder import (
    CompiledKernel,
    LazyTensor,
    kernel,
    to_layout,
    tilize,
    untilize,
    empty,
    zeros,
    full,
    view_layout,
    view,
    permute,
    to_host,
    _REDUCE_SCALER_ARG,
)


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


# --- Block-level elementwise free functions ---------------------------------
#
# Each op wraps the per-tile d2m.tile_* builder in a linalg.generic over a
# tensor of tiles via _eltwise_block. Defined explicitly (rather than via a
# table-driven loop) so the surface is grep- / IDE- / sphinx-discoverable.


@syntax("recip")
def recip(input):
    """Block-level elementwise reciprocal (1/x)."""
    return _eltwise_block(lambda t: d2m.tile_recip(t.type, t), input)


@syntax("exp")
def exp(input):
    """Block-level elementwise exponential."""
    return _eltwise_block(lambda t: d2m.tile_exp(t.type, t), input)


@syntax("log")
def log(input):
    """Block-level elementwise natural logarithm."""
    return _eltwise_block(lambda t: d2m.tile_log(t.type, t), input)


@syntax("negative")
def negative(input):
    """Block-level elementwise negation."""
    return _eltwise_block(lambda t: d2m.tile_negative(t.type, t), input)


@syntax("cos")
def cos(input):
    """Block-level elementwise cosine."""
    return _eltwise_block(lambda t: d2m.tile_cos(t.type, t), input)


@syntax("acos")
def acos(input):
    """Block-level elementwise arccosine."""
    return _eltwise_block(lambda t: d2m.tile_acos(t.type, t), input)


@syntax("sin")
def sin(input):
    """Block-level elementwise sine."""
    return _eltwise_block(lambda t: d2m.tile_sin(t.type, t), input)


@syntax("asin")
def asin(input):
    """Block-level elementwise arcsine."""
    return _eltwise_block(lambda t: d2m.tile_asin(t.type, t), input)


@syntax("tan")
def tan(input):
    """Block-level elementwise tangent."""
    return _eltwise_block(lambda t: d2m.tile_tan(t.type, t), input)


@syntax("atan")
def atan(input):
    """Block-level elementwise arctangent."""
    return _eltwise_block(lambda t: d2m.tile_atan(t.type, t), input)


@syntax("tanh")
def tanh(input):
    """Block-level elementwise hyperbolic tangent."""
    return _eltwise_block(lambda t: d2m.tile_tanh(t.type, t), input)


@syntax("sqrt")
def sqrt(input):
    """Block-level elementwise square root."""
    return _eltwise_block(lambda t: d2m.tile_sqrt(t.type, t), input)


@syntax("rsqrt")
def rsqrt(input):
    """Block-level elementwise reciprocal square root."""
    return _eltwise_block(lambda t: d2m.tile_rsqrt(t.type, t), input)


@syntax("sigmoid")
def sigmoid(input):
    """Block-level elementwise sigmoid."""
    return _eltwise_block(lambda t: d2m.tile_sigmoid(t.type, t), input)


@syntax("hardsigmoid")
def hardsigmoid(input):
    """Block-level elementwise hard-sigmoid."""
    return _eltwise_block(lambda t: d2m.tile_hardsigmoid(t.type, t), input)


@syntax("silu")
def silu(input):
    """Block-level elementwise SiLU (x * sigmoid(x))."""
    return _eltwise_block(lambda t: d2m.tile_silu(t.type, t), input)


@syntax("relu")
def relu(input):
    """Block-level elementwise ReLU."""
    return _eltwise_block(lambda t: d2m.tile_relu(t.type, t), input)


@syntax("gelu")
def gelu(input):
    """Block-level elementwise GELU."""
    return _eltwise_block(lambda t: d2m.tile_gelu(t.type, t), input)


@syntax("erf")
def erf(input):
    """Block-level elementwise error function."""
    return _eltwise_block(lambda t: d2m.tile_erf(t.type, t), input)


@syntax("erfc")
def erfc(input):
    """Block-level elementwise complementary error function."""
    return _eltwise_block(lambda t: d2m.tile_erfc(t.type, t), input)


@syntax("sign")
def sign(input):
    """Block-level elementwise sign."""
    return _eltwise_block(lambda t: d2m.tile_sign(t.type, t), input)


@syntax("ceil")
def ceil(input):
    """Block-level elementwise ceiling."""
    return _eltwise_block(lambda t: d2m.tile_ceil(t.type, t), input)


@syntax("floor")
def floor(input):
    """Block-level elementwise floor."""
    return _eltwise_block(lambda t: d2m.tile_floor(t.type, t), input)


@syntax("abs")
def abs(input):
    """Block-level elementwise absolute value."""
    return _eltwise_block(lambda t: d2m.tile_abs(t.type, t), input)


@syntax("bitwise_not")
def bitwise_not(input):
    """Block-level elementwise bitwise NOT."""
    return _eltwise_block(lambda t: d2m.tile_bitwise_not(t.type, t), input)


@syntax("logical_not")
def logical_not(input):
    """Block-level elementwise logical NOT."""
    return _eltwise_block(lambda t: d2m.tile_logical_not(t.type, t), input)


@syntax("eqz")
def eqz(input):
    """Block-level elementwise is-equal-to-zero predicate."""
    return _eltwise_block(lambda t: d2m.tile_eqz(t.type, t), input)


@syntax("nez")
def nez(input):
    """Block-level elementwise is-not-equal-to-zero predicate."""
    return _eltwise_block(lambda t: d2m.tile_nez(t.type, t), input)


@syntax("gtz")
def gtz(input):
    """Block-level elementwise is-greater-than-zero predicate."""
    return _eltwise_block(lambda t: d2m.tile_gtz(t.type, t), input)


@syntax("gez")
def gez(input):
    """Block-level elementwise is-greater-than-or-equal-to-zero predicate."""
    return _eltwise_block(lambda t: d2m.tile_gez(t.type, t), input)


@syntax("ltz")
def ltz(input):
    """Block-level elementwise is-less-than-zero predicate."""
    return _eltwise_block(lambda t: d2m.tile_ltz(t.type, t), input)


@syntax("lez")
def lez(input):
    """Block-level elementwise is-less-than-or-equal-to-zero predicate."""
    return _eltwise_block(lambda t: d2m.tile_lez(t.type, t), input)


@syntax("exp2")
def exp2(input):
    """Block-level elementwise base-2 exponential (2**x)."""
    return _eltwise_block(lambda t: d2m.tile_exp2(t.type, t), input)


@syntax("expm1")
def expm1(input):
    """Block-level elementwise exp(x) - 1 (numerically stable for small x)."""
    return _eltwise_block(lambda t: d2m.tile_expm1(t.type, t), input)


@syntax("log1p")
def log1p(input):
    """Block-level elementwise log(1 + x) (numerically stable for small x)."""
    return _eltwise_block(lambda t: d2m.tile_log1p(t.type, t), input)


@syntax("square")
def square(input):
    """Block-level elementwise square (x * x)."""
    return _eltwise_block(lambda t: d2m.tile_square(t.type, t), input)


@syntax("softsign")
def softsign(input):
    """Block-level elementwise softsign (x / (1 + |x|))."""
    return _eltwise_block(lambda t: d2m.tile_softsign(t.type, t), input)


@syntax("selu")
def selu(input):
    """Block-level elementwise scaled exponential linear unit."""
    return _eltwise_block(lambda t: d2m.tile_selu(t.type, t), input)


@syntax("signbit")
def signbit(input):
    """Block-level elementwise IEEE-754 sign bit (0.0 or 1.0)."""
    return _eltwise_block(lambda t: d2m.tile_signbit(t.type, t), input)


@syntax("frac")
def frac(input):
    """Block-level elementwise fractional part."""
    return _eltwise_block(lambda t: d2m.tile_frac(t.type, t), input)


@syntax("trunc")
def trunc(input):
    """Block-level elementwise truncate toward zero."""
    return _eltwise_block(lambda t: d2m.tile_trunc(t.type, t), input)


@syntax("add")
def add(lhs, rhs):
    """Block-level elementwise addition."""
    return _eltwise_block(lambda l, r: d2m.tile_add(l.type, l, r), lhs, rhs)


@syntax("sub")
def sub(lhs, rhs):
    """Block-level elementwise subtraction."""
    return _eltwise_block(lambda l, r: d2m.tile_sub(l.type, l, r), lhs, rhs)


@syntax("mul")
def mul(lhs, rhs):
    """Block-level elementwise multiplication."""
    return _eltwise_block(lambda l, r: d2m.tile_mul(l.type, l, r), lhs, rhs)


@syntax("div")
def div(lhs, rhs):
    """Block-level elementwise division."""
    return _eltwise_block(lambda l, r: d2m.tile_div(l.type, l, r), lhs, rhs)


@syntax("pow")
def pow(lhs, rhs):
    """Block-level elementwise exponentiation (a ** b)."""
    return _eltwise_block(lambda l, r: d2m.tile_pow(l.type, l, r), lhs, rhs)


@syntax("maximum")
def maximum(lhs, rhs):
    """Block-level elementwise elementwise maximum."""
    return _eltwise_block(lambda l, r: d2m.tile_maximum(l.type, l, r), lhs, rhs)


@syntax("minimum")
def minimum(lhs, rhs):
    """Block-level elementwise elementwise minimum."""
    return _eltwise_block(lambda l, r: d2m.tile_minimum(l.type, l, r), lhs, rhs)


@syntax("bitwise_and")
def bitwise_and(lhs, rhs):
    """Block-level elementwise bitwise AND."""
    return _eltwise_block(lambda l, r: d2m.tile_bitwise_and(l.type, l, r), lhs, rhs)


@syntax("bitwise_or")
def bitwise_or(lhs, rhs):
    """Block-level elementwise bitwise OR."""
    return _eltwise_block(lambda l, r: d2m.tile_bitwise_or(l.type, l, r), lhs, rhs)


@syntax("bitwise_xor")
def bitwise_xor(lhs, rhs):
    """Block-level elementwise bitwise XOR."""
    return _eltwise_block(lambda l, r: d2m.tile_bitwise_xor(l.type, l, r), lhs, rhs)


@syntax("logical_left_shift")
def logical_left_shift(lhs, rhs):
    """Block-level elementwise logical (zero-fill) left shift."""
    return _eltwise_block(
        lambda l, r: d2m.tile_logical_left_shift(l.type, l, r), lhs, rhs
    )


@syntax("logical_right_shift")
def logical_right_shift(lhs, rhs):
    """Block-level elementwise logical (zero-fill) right shift."""
    return _eltwise_block(
        lambda l, r: d2m.tile_logical_right_shift(l.type, l, r), lhs, rhs
    )


@syntax("right_shift")
def right_shift(lhs, rhs):
    """Block-level elementwise arithmetic (sign-preserving) right shift."""
    return _eltwise_block(lambda l, r: d2m.tile_right_shift(l.type, l, r), lhs, rhs)


@syntax("where")
def where(cond, true_value, false_value):
    """Block-level elementwise select: `cond ? true_value : false_value`.

    All three blocks must have the same type. `cond` is interpreted
    elementwise as a boolean -- non-zero selects `true_value`, zero
    selects `false_value`."""
    return _eltwise_block(
        lambda c, t, f: d2m.tile_where(t.type, c, t, f),
        cond,
        true_value,
        false_value,
    )


@syntax("matmul")
def matmul(lhs, rhs):
    """Block-level matmul: `C = A @ B` (see _matmul_block)."""
    return _matmul_block(lhs, rhs)


@syntax("reduce_sum")
def reduce_sum(input, dim):
    """Block-level float sum reduction over one tile axis.

    `dim` follows torch/numpy axis numbering for a 2D tile block:
    `0` reduces rows and `1` reduces columns.
    """
    return _reduce_block(
        lambda a, b, c, reduce_dim: d2m.tile_reduce_sum(a.type, a, b, c, reduce_dim),
        input,
        dim,
        0.0,
    )


@syntax("reduce_max")
def reduce_max(input, dim):
    """Block-level float max reduction over one tile axis.

    `dim` follows torch/numpy axis numbering for a 2D tile block:
    `0` reduces rows and `1` reduces columns.
    """
    return _reduce_block(
        lambda a, b, c, reduce_dim: d2m.tile_reduce_max(a.type, a, b, c, reduce_dim),
        input,
        dim,
        float("-inf"),
    )


@syntax("!tensor")
class TensorBlock:
    """The DSL-side host class for a tile-typed tensor block.

    Python operator dunders (__add__, __neg__, ...) are dispatched by
    D2MCompiler.visit_BinOp / visit_UnaryOp via '!tensor.__add__' etc.
    Method-style ops (block.exp(), block.add(other), ...) are dispatched by
    visit_Attribute via '!tensor.exp' etc. Both paths delegate to the
    module-level free functions defined above.
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    # --- Python operator dunders ----
    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return add(ast_self, rhs)

    def __sub__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return sub(ast_self, rhs)

    def __mul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return mul(ast_self, rhs)

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return div(ast_self, rhs)

    def __neg__(ast_self: TensorBlock) -> TensorBlock:
        return negative(ast_self)

    def __invert__(ast_self: TensorBlock) -> TensorBlock:
        return bitwise_not(ast_self)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return matmul(ast_self, rhs)

    # --- Unary block-level method forms ----
    def recip(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.recip(self)`."""
        return recip(ast_self)

    def exp(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.exp(self)`."""
        return exp(ast_self)

    def log(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.log(self)`."""
        return log(ast_self)

    def negative(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.negative(self)`."""
        return negative(ast_self)

    def cos(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.cos(self)`."""
        return cos(ast_self)

    def acos(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.acos(self)`."""
        return acos(ast_self)

    def sin(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.sin(self)`."""
        return sin(ast_self)

    def asin(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.asin(self)`."""
        return asin(ast_self)

    def tan(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.tan(self)`."""
        return tan(ast_self)

    def atan(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.atan(self)`."""
        return atan(ast_self)

    def tanh(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.tanh(self)`."""
        return tanh(ast_self)

    def sqrt(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.sqrt(self)`."""
        return sqrt(ast_self)

    def rsqrt(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.rsqrt(self)`."""
        return rsqrt(ast_self)

    def sigmoid(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.sigmoid(self)`."""
        return sigmoid(ast_self)

    def hardsigmoid(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.hardsigmoid(self)`."""
        return hardsigmoid(ast_self)

    def silu(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.silu(self)`."""
        return silu(ast_self)

    def relu(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.relu(self)`."""
        return relu(ast_self)

    def gelu(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.gelu(self)`."""
        return gelu(ast_self)

    def erf(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.erf(self)`."""
        return erf(ast_self)

    def erfc(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.erfc(self)`."""
        return erfc(ast_self)

    def sign(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.sign(self)`."""
        return sign(ast_self)

    def ceil(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.ceil(self)`."""
        return ceil(ast_self)

    def floor(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.floor(self)`."""
        return floor(ast_self)

    def abs(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.abs(self)`."""
        return abs(ast_self)

    def bitwise_not(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.bitwise_not(self)`."""
        return bitwise_not(ast_self)

    def logical_not(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.logical_not(self)`."""
        return logical_not(ast_self)

    def eqz(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.eqz(self)`."""
        return eqz(ast_self)

    def nez(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.nez(self)`."""
        return nez(ast_self)

    def gtz(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.gtz(self)`."""
        return gtz(ast_self)

    def gez(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.gez(self)`."""
        return gez(ast_self)

    def ltz(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.ltz(self)`."""
        return ltz(ast_self)

    def lez(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.lez(self)`."""
        return lez(ast_self)

    def exp2(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.exp2(self)`."""
        return exp2(ast_self)

    def expm1(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.expm1(self)`."""
        return expm1(ast_self)

    def log1p(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.log1p(self)`."""
        return log1p(ast_self)

    def square(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.square(self)`."""
        return square(ast_self)

    def softsign(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.softsign(self)`."""
        return softsign(ast_self)

    def selu(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.selu(self)`."""
        return selu(ast_self)

    def signbit(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.signbit(self)`."""
        return signbit(ast_self)

    def frac(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.frac(self)`."""
        return frac(ast_self)

    def trunc(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.trunc(self)`."""
        return trunc(ast_self)

    # --- Binary block-level method forms ----
    def add(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.add(self, rhs)`."""
        return add(ast_self, rhs)

    def sub(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.sub(self, rhs)`."""
        return sub(ast_self, rhs)

    def mul(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.mul(self, rhs)`."""
        return mul(ast_self, rhs)

    def div(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.div(self, rhs)`."""
        return div(ast_self, rhs)

    def pow(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.pow(self, rhs)`."""
        return pow(ast_self, rhs)

    def maximum(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.maximum(self, rhs)`."""
        return maximum(ast_self, rhs)

    def minimum(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.minimum(self, rhs)`."""
        return minimum(ast_self, rhs)

    def bitwise_and(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.bitwise_and(self, rhs)`."""
        return bitwise_and(ast_self, rhs)

    def bitwise_or(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.bitwise_or(self, rhs)`."""
        return bitwise_or(ast_self, rhs)

    def bitwise_xor(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.bitwise_xor(self, rhs)`."""
        return bitwise_xor(ast_self, rhs)

    def logical_left_shift(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.logical_left_shift(self, rhs)`."""
        return logical_left_shift(ast_self, rhs)

    def logical_right_shift(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.logical_right_shift(self, rhs)`."""
        return logical_right_shift(ast_self, rhs)

    def right_shift(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.right_shift(self, rhs)`."""
        return right_shift(ast_self, rhs)

    def where(ast_self: TensorBlock, true_value, false_value) -> TensorBlock:
        """Same as `d2m.where(self, true_value, false_value)` -- `self` is
        the condition tile."""
        return where(ast_self, true_value, false_value)

    def matmul(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.matmul(self, rhs)`."""
        return matmul(ast_self, rhs)

    def reduce_sum(ast_self: TensorBlock, dim) -> TensorBlock:
        """Same as `d2m.reduce_sum(self, dim)`."""
        return reduce_sum(ast_self, dim)

    def reduce_max(ast_self: TensorBlock, dim) -> TensorBlock:
        """Same as `d2m.reduce_max(self, dim)`."""
        return reduce_max(ast_self, dim)

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return d2m.store(ast_self, rhs)


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


# --- Block-level eltwise helper -------------------------------------------


def _eltwise_block(tile_op_fn, *blocks):
    """Wrap an N-ary per-tile op inside a `linalg.generic` over tensors of
    tiles.

    All `blocks` must have the same tensor type. `tile_op_fn` is called
    with the scalar (tile-typed) values for each input and must return
    the scalar output value (or an OpView whose .result is the value).

    Emits:
        %out = d2m.empty() : tensor<...x!ttcore.tile<...>>
        %ret = linalg.generic
            { indexing_maps = [identity] * (N+1),
              iterator_types = [parallel] * rank }
            ins(%b0, ..., %b_{N-1}) outs(%out) {
          ^bb0(%t0: !tile, ..., %t_{N-1}: !tile, %t_out: !tile):
            %r = tile_op_fn(%t0, ..., %t_{N-1})
            linalg.yield %r : !tile
        } -> tensor<...x!ttcore.tile<...>>
    """
    if not blocks:
        raise ValueError("_eltwise_block requires at least one input block")
    block_ty = blocks[0].type
    for b in blocks[1:]:
        if b.type != block_ty:
            raise ValueError(
                f"_eltwise_block: input block type mismatch: {b.type} vs {block_ty}"
            )
    rank = block_ty.rank
    elem_ty = block_ty.element_type

    output = d2m.empty(block_ty)
    identity = AffineMap.get_identity(rank)
    n_args = len(blocks) + 1  # one input map per block + one output map
    indexing_maps = ArrayAttr.get([AffineMapAttr.get(identity)] * n_args)
    parallel = Attribute.parse("#linalg.iterator_type<parallel>")
    iterator_types = ArrayAttr.get([parallel] * rank)

    generic = linalg.GenericOp(
        [block_ty],
        list(blocks),
        [output],
        indexing_maps,
        iterator_types,
    )
    body_arg_tys = [elem_ty] * n_args
    body_arg_locs = [Location.unknown()] * n_args
    body = Block.create_at_start(generic.regions[0], body_arg_tys, body_arg_locs)
    with InsertionPoint(body):
        result = tile_op_fn(*body.arguments[: len(blocks)])
        if hasattr(result, "result"):
            result = result.result
        linalg.yield_([result])
    return generic.result


def _dim_to_reduce_dim_attr(dim):
    if isinstance(dim, arith.ConstantOp):
        dim = IntegerAttr(dim.value).value
    elif isinstance(dim, IntegerAttr):
        dim = dim.value
    elif isinstance(dim, Attribute):
        return dim

    if dim in (0, -2):
        return Attribute.parse("#d2m<reduce_dim C>")
    if dim in (1, -1):
        return Attribute.parse("#d2m<reduce_dim R>")
    raise ValueError(f"reduce dim must be 0/1 or -2/-1, got {dim}")


def _float_scalar_type_for_tile(tile_type):
    ctx = tile_type.context
    tile_type = ttcore.ir.TileType.maybe_downcast(tile_type)
    if tile_type is None:
        raise TypeError(f"expected a ttcore.tile element type, got {tile_type}")
    data_type = tile_type.data_type_as_int
    if data_type == int(ttcore.DataType.Float32):
        return F32Type.get(ctx)
    if data_type == int(ttcore.DataType.Float16):
        return F16Type.get(ctx)
    if data_type == int(ttcore.DataType.BFloat16):
        return BF16Type.get(ctx)
    raise TypeError(
        "float reductions require f32, f16, or bf16 tile element types; "
        f"got {tile_type}"
    )


def _tile_fill_float(tile_type, value):
    scalar_type = _float_scalar_type_for_tile(tile_type)
    scalar_attr = FloatAttr.get(scalar_type, value)
    scalar = arith.ConstantOp(scalar_type, scalar_attr).result
    return d2m.tile_fill(tile_type, scalar)


def _get_reduce_scaler_block(input_block):
    compiler = D2MCompiler.current()
    if compiler is None:
        raise RuntimeError("reduce_sum/reduce_max can only be used inside @kernel")

    key = (str(input_block.type.element_type), id(InsertionPoint.current.block))
    cached = compiler.reduce_scaler_cache.get(key)
    if cached is not None:
        return cached

    scaler = compiler.get_synthetic_arg(_REDUCE_SCALER_ARG)
    if scaler is None:
        raise RuntimeError("internal error: missing synthetic reduction scaler")

    zero = arith.ConstantOp(IndexType.get(scaler.context), 0).result
    scaler_block = remote_load(scaler, [zero, zero])
    if scaler_block.type.element_type != input_block.type.element_type:
        raise TypeError(
            "reduction scaler tile type mismatch: "
            f"{scaler_block.type.element_type} vs {input_block.type.element_type}"
        )
    compiler.reduce_scaler_cache[key] = scaler_block
    return scaler_block


def _reduce_block(tile_op_fn, input, dim, identity_value):
    """Wrap a float d2m.tile_reduce_* op in a per-block linalg.generic.

    The generated result has the same block shape as `input`. The reduction is
    within each tile; lanes outside the reduced row/column follow the hardware
    tile_reduce_* result layout.
    """
    block_ty = input.type
    if not isinstance(block_ty, RankedTensorType):
        raise TypeError(f"reduce input must be a ranked tensor, got {block_ty}")

    rank = block_ty.rank
    elem_ty = block_ty.element_type
    reduce_dim = _dim_to_reduce_dim_attr(dim)
    scaler = _get_reduce_scaler_block(input)

    output = d2m.empty(block_ty)
    identity = AffineMap.get_identity(rank)
    zero = AffineConstantExpr.get(0)
    scaler_map = AffineMap.get(rank, 0, [zero, zero])
    indexing_maps = ArrayAttr.get(
        [
            AffineMapAttr.get(identity),
            AffineMapAttr.get(scaler_map),
            AffineMapAttr.get(identity),
        ]
    )
    parallel = Attribute.parse("#linalg.iterator_type<parallel>")
    iterator_types = ArrayAttr.get([parallel] * rank)

    generic = linalg.GenericOp(
        [block_ty],
        [input, scaler],
        [output],
        indexing_maps,
        iterator_types,
    )
    body = Block.create_at_start(
        generic.regions[0],
        [elem_ty, elem_ty, elem_ty],
        [Location.unknown()] * 3,
    )
    with InsertionPoint(body):
        input_tile, scaler_tile, _ = body.arguments
        accumulator = _tile_fill_float(elem_ty, identity_value)
        result = tile_op_fn(input_tile, scaler_tile, accumulator, reduce_dim)
        if hasattr(result, "result"):
            result = result.result
        linalg.yield_([result])
    return generic.result


def _matmul_block(lhs, rhs):
    """Block-level matmul: `C = A @ B` where each tensor is a 2D block of
    tiles. Emits a linalg.generic with the standard matmul indexing maps
    (parallel/parallel/reduction over M/N/K) and `d2m.tile_matmul` in the
    body (per-tile accumulating multiply-add).

    lhs: tensor<M x K x !ttcore.tile<...>>
    rhs: tensor<K x N x !ttcore.tile<...>>
    Returns: tensor<M x N x !ttcore.tile<...>>.
    """
    assert isinstance(lhs.type, RankedTensorType)
    assert isinstance(rhs.type, RankedTensorType)
    assert lhs.type.rank == 2, f"matmul lhs must be 2D, got rank {lhs.type.rank}"
    assert rhs.type.rank == 2, f"matmul rhs must be 2D, got rank {rhs.type.rank}"
    assert lhs.type.element_type == rhs.type.element_type, (
        f"matmul element type mismatch: {lhs.type.element_type} vs "
        f"{rhs.type.element_type}"
    )
    elem_ty = lhs.type.element_type
    m_blocks = lhs.type.shape[0]
    k_blocks = lhs.type.shape[1]
    assert (
        k_blocks == rhs.type.shape[0]
    ), f"matmul inner dim mismatch: lhs K={k_blocks} vs rhs K={rhs.type.shape[0]}"
    n_blocks = rhs.type.shape[1]

    out_ty = RankedTensorType.get([m_blocks, n_blocks], elem_ty)
    # TODO: zero-initialise the accumulator. The matmul body computes
    # `c = c + a @ b`, so an uninitialised output yields garbage. A
    # host-scope linalg.generic-as-fill (the natural way to express this
    # at this layer) does not survive the d2m -> ttkernel conversion
    # today. Users needing correct matmul should pre-fill the output via
    # `d2m.zeros(L)` and pass that as the out-param to a kernel that
    # calls `@`.
    output = d2m.empty(out_ty)

    # (d0, d1, d2) = (M, N, K).
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    d2 = AffineDimExpr.get(2)
    a_map = AffineMap.get(3, 0, [d0, d2])
    b_map = AffineMap.get(3, 0, [d2, d1])
    c_map = AffineMap.get(3, 0, [d0, d1])
    indexing_maps = ArrayAttr.get(
        [
            AffineMapAttr.get(a_map),
            AffineMapAttr.get(b_map),
            AffineMapAttr.get(c_map),
        ]
    )
    parallel = Attribute.parse("#linalg.iterator_type<parallel>")
    reduction = Attribute.parse("#linalg.iterator_type<reduction>")
    iterator_types = ArrayAttr.get([parallel, parallel, reduction])

    generic = linalg.GenericOp(
        [out_ty],
        [lhs, rhs],
        [output],
        indexing_maps,
        iterator_types,
    )
    body = Block.create_at_start(
        generic.regions[0],
        [elem_ty, elem_ty, elem_ty],
        [Location.unknown()] * 3,
    )
    with InsertionPoint(body):
        a_t, b_t, c_t = body.arguments
        result = d2m.tile_matmul(c_t.type, a_t, b_t, c_t)
        if hasattr(result, "result"):
            result = result.result
        linalg.yield_([result])
    return generic.result
