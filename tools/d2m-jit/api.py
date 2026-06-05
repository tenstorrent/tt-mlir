# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast

from ttmlir.ir import *
from ttmlir.dialects import d2m, ttcore, arith, linalg

from ._src.utils import _asindex
from ._src.ast import D2MCompiler, syntax
from ._src.config import config
from ._src.errors import D2mJitError
from ._src.tensor_layout import Layout, float32, float16, bfloat16, _to_data_type
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
    arange,
    view_layout,
    view,
    permute,
    reshape,
    to_host,
)
from ._src.rewrite import (
    pattern,
    from_value,
    from_device,
    infer_layout,
    apply_patterns,
)

TileBcastType = d2m.TileBcastType


def _parse_tile_bcast_type(value):
    if isinstance(value, Attribute):
        return value
    if isinstance(value, d2m.TileBcastType):
        value = int(value)
    if isinstance(value, str):
        key = value.lower()
        if key in {"none", "no_bcast"}:
            return Attribute.parse("#d2m<tile_bcast_type none>")
        if key in {"col", "column"}:
            return Attribute.parse("#d2m<tile_bcast_type col>")
        if key == "row":
            return Attribute.parse("#d2m<tile_bcast_type row>")
        if key == "2d":
            return Attribute.parse("#d2m<tile_bcast_type scalar>")
    if isinstance(value, int) and not isinstance(value, bool):
        if value == int(d2m.TileBcastType.None_):
            return Attribute.parse("#d2m<tile_bcast_type none>")
        if value == int(d2m.TileBcastType.Col):
            return Attribute.parse("#d2m<tile_bcast_type col>")
        if value == int(d2m.TileBcastType.Row):
            return Attribute.parse("#d2m<tile_bcast_type row>")
        if value == int(d2m.TileBcastType.Scalar):
            return Attribute.parse("#d2m<tile_bcast_type scalar>")
    raise ValueError(
        "tile broadcast type must be one of 'row', 'col', '2d', 'none', "
        f"or a d2m.TileBcastType, got {value!r}"
    )


def _tile_bcast_type_attr(node):
    return _parse_tile_bcast_type(node.value)


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


# --- Bespoke-signature unary ops --------------------------------------------
#
# These take additional Python-literal arguments that lower to MLIR
# attributes on the tile op (rather than runtime values). When called from
# inside a `@d2m.kernel` body, the literal args are pulled out of the AST
# via `_const_value`; from regular Python the wrapper accepts the same
# literals directly.


def _const_value(node):
    """args_as_attr callback: pull a Python literal out of `node`.

    Accepts a bare `ast.Constant`, or a `+literal`/`-literal` UnaryOp on
    a numeric constant (Python parses `-0.5` as `UnaryOp(USub,
    Constant(0.5))`). Raises a clear error otherwise so kernel users
    get a useful message instead of an obscure compile-time crash."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)):
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
    raise D2mJitError(
        f"expected a Python literal (number/string), got "
        f"{type(node).__name__}; runtime values are not supported "
        f"for attribute-typed kernel arguments"
    )


def _tile_elem_type(block):
    """Return the !ttcore.tile element type of a tile-tensor block, or
    raise a clear error if `block` isn't a tile tensor."""
    elem_ty = ttcore.ir.TileType.maybe_downcast(block.type.element_type)
    if elem_ty is None:
        raise TypeError(
            f"expected a tile-typed tensor block, got element type "
            f"{block.type.element_type}"
        )
    return elem_ty


# The ttcore.DataType returned by `TileType.data_type` comes from a
# different python-binding module than `ttcore.DataType.Float32` (separate
# nanobind modules each bind the same C++ enum), so `==` and `in` fail.
# Compare via `.name` strings.
_FLOAT_DATA_TYPE_NAMES = frozenset({"Float32", "Float16", "BFloat16"})


def _is_float_tile_dtype(tile_elem_ty):
    return tile_elem_ty.data_type.name in _FLOAT_DATA_TYPE_NAMES


@syntax(
    "clamp_scalar",
    args_as_attr=[False, _const_value, _const_value],
)
def clamp_scalar(input, min, max):
    """Block-level elementwise clamp to scalar bounds `[min, max]`.

    `min` and `max` must be Python literals. The attribute type is picked
    from the tile element type: float tiles use F32Attr; integer tiles use
    I32Attr (the verifier requires both bounds to be the same attr type).
    """
    elem_ty = _tile_elem_type(input)
    if _is_float_tile_dtype(elem_ty):
        f32 = F32Type.get()
        min_attr = FloatAttr.get(f32, float(min))
        max_attr = FloatAttr.get(f32, float(max))
    else:
        i32 = IntegerType.get_signless(32)
        min_attr = IntegerAttr.get(i32, int(min))
        max_attr = IntegerAttr.get(i32, int(max))
    return _eltwise_block(
        lambda t: d2m.tile_clamp_scalar(t.type, t, min_attr, max_attr),
        input,
    )


@syntax(
    "typecast",
    args_as_attr=[False, _const_value],
)
def typecast(input, dtype):
    """Block-level elementwise per-tile typecast to `dtype`.

    `dtype` is a d2m DataType (e.g. `d2m.bfloat16`) or one of the strings
    accepted by `_to_data_type` (`"fp32"`, `"bf16"`, `"fp16"`, ...). The
    output block has the same shape as the input but a different tile
    element type.

    Distinct from host-side `tilize(dtype=...)` / `untilize(dtype=...)`:
    those convert a `LazyTensor`'s layout off-device; this one happens
    inside the kernel body during compute.
    """
    return _typecast_block(input, dtype)


@syntax("tile_transpose")
def tile_transpose(input):
    """Block-level elementwise per-tile (32x32) transpose.

    Distinct from logical `d2m.permute` / `d2m.view` (host-side layout
    transformations on a `LazyTensor`): this one transposes each tile
    in-place inside a kernel body.
    """
    return _eltwise_block(lambda t: d2m.tile_transpose(t.type, t), input)


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


# Binary comparisons. The result tile element type matches the input -- the
# comparison primitive writes 1.0/0.0 (or 1/0 for int tiles) into each lane,
# matching the dialect's tile_eqz / tile_gez / ... family. Pair with `where`
# to build masked computations.
#
# Not wired into Python `<` / `>=` / `==` dunders: visit_Compare in the AST
# visitor currently lowers compare ops to `arith.cmpi` for index-domain
# conditions (scf loop bounds, etc.), so overloading the dunders would
# conflict. Use the free-function or method forms instead: `d2m.ge(a, b)`
# or `a.ge(b)`.


@syntax("eq")
def eq(lhs, rhs):
    """Block-level elementwise equality: `lhs == rhs`."""
    return _eltwise_block(lambda l, r: d2m.tile_eq(l.type, l, r), lhs, rhs)


@syntax("ne")
def ne(lhs, rhs):
    """Block-level elementwise inequality: `lhs != rhs`."""
    return _eltwise_block(lambda l, r: d2m.tile_ne(l.type, l, r), lhs, rhs)


@syntax("gt")
def gt(lhs, rhs):
    """Block-level elementwise greater-than: `lhs > rhs`."""
    return _eltwise_block(lambda l, r: d2m.tile_gt(l.type, l, r), lhs, rhs)


@syntax("ge")
def ge(lhs, rhs):
    """Block-level elementwise greater-than-or-equal: `lhs >= rhs`."""
    return _eltwise_block(lambda l, r: d2m.tile_ge(l.type, l, r), lhs, rhs)


@syntax("lt")
def lt(lhs, rhs):
    """Block-level elementwise less-than: `lhs < rhs`."""
    return _eltwise_block(lambda l, r: d2m.tile_lt(l.type, l, r), lhs, rhs)


@syntax("le")
def le(lhs, rhs):
    """Block-level elementwise less-than-or-equal: `lhs <= rhs`."""
    return _eltwise_block(lambda l, r: d2m.tile_le(l.type, l, r), lhs, rhs)


@syntax("tile_bcast", args_as_attr=[False, _tile_bcast_type_attr])
def tile_bcast(input, bcast_type):
    """Block-level tile broadcast.

    `bcast_type` matches the D2M tile-broadcast enum: `row` broadcasts the
    tile's 0-row, `col` broadcasts the tile's 0-column, and `scalar`
    broadcasts element (0, 0). The result has the same block shape as the
    input, with each tile expanded according to `bcast_type`.
    """
    tile_bcast_type = _parse_tile_bcast_type(bcast_type)
    return _eltwise_block(lambda t: d2m.tile_bcast(t.type, t, tile_bcast_type), input)


@syntax("tile_bcast_row")
def tile_bcast_row(input):
    """Block-level tile broadcast of the tile's 0-row."""
    return tile_bcast(input, d2m.TileBcastType.Row)


@syntax("tile_bcast_col")
def tile_bcast_col(input):
    """Block-level tile broadcast of the tile's 0-column."""
    return tile_bcast(input, d2m.TileBcastType.Col)


@syntax("tile_bcast_2d")
def tile_bcast_2d(input):
    """Block-level tile broadcast of element (0, 0) across rows and columns."""
    return tile_bcast(input, d2m.TileBcastType.Scalar)


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

    def tile_transpose(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.tile_transpose(self)` -- per-tile (32x32) transpose."""
        return tile_transpose(ast_self)

    # NOTE: clamp_scalar / typecast / bcast intentionally have no
    # method-style form. They carry Python-literal attribute arguments,
    # and the AST visitor's `args_as_attr` mechanism only kicks in on
    # free-function calls (visit_Call on a Name target). Method calls go
    # through visit_Attribute and fully evaluate their args, which would
    # fall over on float literals (visit_Constant doesn't handle floats).
    # Use the free-function form: `clamp_scalar(x, -1.0, 1.0)` etc.

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

    def eq(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.eq(self, rhs)`."""
        return eq(ast_self, rhs)

    def ne(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.ne(self, rhs)`."""
        return ne(ast_self, rhs)

    def gt(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.gt(self, rhs)`."""
        return gt(ast_self, rhs)

    def ge(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.ge(self, rhs)`."""
        return ge(ast_self, rhs)

    def lt(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.lt(self, rhs)`."""
        return lt(ast_self, rhs)

    def le(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.le(self, rhs)`."""
        return le(ast_self, rhs)

    def tile_bcast_row(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.tile_bcast_row(self)`."""
        return tile_bcast_row(ast_self)

    def tile_bcast_col(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.tile_bcast_col(self)`."""
        return tile_bcast_col(ast_self)

    def tile_bcast_2d(ast_self: TensorBlock) -> TensorBlock:
        """Same as `d2m.tile_bcast_2d(self)`."""
        return tile_bcast_2d(ast_self)

    def where(ast_self: TensorBlock, true_value, false_value) -> TensorBlock:
        """Same as `d2m.where(self, true_value, false_value)` -- `self` is
        the condition tile."""
        return where(ast_self, true_value, false_value)

    def matmul(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Same as `d2m.matmul(self, rhs)`."""
        return matmul(ast_self, rhs)

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


def _typecast_block(input, dtype):
    """Tile-level typecast inside a linalg.generic. Output block has the
    same shape as `input` but a different tile element type.

    Separate from `_eltwise_block` because that helper assumes input and
    output blocks share an element type; typecast deliberately changes it.
    """
    src_tile_dc = _tile_elem_type(input)
    dst_data_type = _to_data_type(dtype)
    if src_tile_dc.data_type.name == dst_data_type.name:
        # No-op: avoid emitting a typecast for a same-dtype cast (the
        # backend may not support the resulting init/run pair).
        return input
    src_shape = src_tile_dc.shape
    # `dst_tile_ty` is a plain ir.Type; `block_ty.element_type` is also a
    # plain ir.Type. We deliberately use those (not the downcast
    # `tt_ir.TileType`) as Block arg types -- Block.create_at_start does
    # not accept the downcast subclass.
    dst_tile_ty = ttcore.ir.TileType.get(
        input.context, src_shape[0], src_shape[1], dst_data_type
    )
    block_ty = input.type
    src_tile_ty = block_ty.element_type
    rank = block_ty.rank
    out_block_ty = RankedTensorType.get(list(block_ty.shape), dst_tile_ty)

    output = d2m.empty(out_block_ty)
    identity = AffineMap.get_identity(rank)
    indexing_maps = ArrayAttr.get([AffineMapAttr.get(identity)] * 2)
    parallel = Attribute.parse("#linalg.iterator_type<parallel>")
    iterator_types = ArrayAttr.get([parallel] * rank)

    generic = linalg.GenericOp(
        [out_block_ty],
        [input],
        [output],
        indexing_maps,
        iterator_types,
    )
    body = Block.create_at_start(
        generic.regions[0],
        [src_tile_ty, dst_tile_ty],
        [Location.unknown()] * 2,
    )
    with InsertionPoint(body):
        casted = d2m.tile_typecast(dst_tile_ty, body.arguments[0])
        if hasattr(casted, "result"):
            casted = casted.result
        linalg.yield_([casted])
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
