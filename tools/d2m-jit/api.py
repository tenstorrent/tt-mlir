# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import *
from ttmlir.dialects import d2m, arith, linalg

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


class TensorBlock:
    """The DSL-side host class for a tile-typed tensor block. Methods are
    populated below from _UNARY_OPS / _BINARY_OPS tables, and the class is
    registered into D2MCompiler._syntax via syntax("!tensor") afterwards."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    # Python operator dispatch. visit_BinOp/visit_UnaryOp in D2MCompiler
    # look these up under '!tensor.__add__', '!tensor.__neg__', etc.
    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return _eltwise_block(lambda l, r: d2m.tile_add(l.type, l, r), ast_self, rhs)

    def __sub__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return _eltwise_block(lambda l, r: d2m.tile_sub(l.type, l, r), ast_self, rhs)

    def __mul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return _eltwise_block(lambda l, r: d2m.tile_mul(l.type, l, r), ast_self, rhs)

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return _eltwise_block(lambda l, r: d2m.tile_div(l.type, l, r), ast_self, rhs)

    def __neg__(ast_self: TensorBlock) -> TensorBlock:
        return _eltwise_block(lambda t: d2m.tile_negative(t.type, t), ast_self)

    def __invert__(ast_self: TensorBlock) -> TensorBlock:
        return _eltwise_block(lambda t: d2m.tile_bitwise_not(t.type, t), ast_self)

    # @ (matmul) stays bespoke -- not elementwise.
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


# --- Block-level elementwise op tables --------------------------------------
#
# Every entry below registers both a standalone @syntax(name) free function
# and a TensorBlock method, both wired through _eltwise_block. The d2m.tile_*
# builders are wrapped in a linalg.generic so the DSL operates on
# tensor<...x!ttcore.tile<...>> values (blocks of tiles) while the d2m ops
# themselves still see scalar tile operands inside the linalg body.

_UNARY_OPS = {
    "recip":       d2m.tile_recip,
    "exp":         d2m.tile_exp,
    "log":         d2m.tile_log,
    "negative":    d2m.tile_negative,
    "cos":         d2m.tile_cos,
    "acos":        d2m.tile_acos,
    "sin":         d2m.tile_sin,
    "asin":        d2m.tile_asin,
    "tan":         d2m.tile_tan,
    "atan":        d2m.tile_atan,
    "tanh":        d2m.tile_tanh,
    "sqrt":        d2m.tile_sqrt,
    "rsqrt":       d2m.tile_rsqrt,
    "sigmoid":     d2m.tile_sigmoid,
    "hardsigmoid": d2m.tile_hardsigmoid,
    "silu":        d2m.tile_silu,
    "relu":        d2m.tile_relu,
    "gelu":        d2m.tile_gelu,
    "erf":         d2m.tile_erf,
    "erfc":        d2m.tile_erfc,
    "sign":        d2m.tile_sign,
    "ceil":        d2m.tile_ceil,
    "floor":       d2m.tile_floor,
    "abs":         d2m.tile_abs,
    "bitwise_not": d2m.tile_bitwise_not,
    "logical_not": d2m.tile_logical_not,
    "eqz":         d2m.tile_eqz,
    "nez":         d2m.tile_nez,
    "gtz":         d2m.tile_gtz,
    "gez":         d2m.tile_gez,
    "ltz":         d2m.tile_ltz,
    "lez":         d2m.tile_lez,
}

_BINARY_OPS = {
    "add":         d2m.tile_add,
    "sub":         d2m.tile_sub,
    "mul":         d2m.tile_mul,
    "div":         d2m.tile_div,
    "pow":         d2m.tile_pow,
    "maximum":     d2m.tile_maximum,
    "minimum":     d2m.tile_minimum,
    "bitwise_and": d2m.tile_bitwise_and,
    "bitwise_or":  d2m.tile_bitwise_or,
    "bitwise_xor": d2m.tile_bitwise_xor,
}


def _make_unary(builder):
    def fn(ast_self):
        return _eltwise_block(lambda t: builder(t.type, t), ast_self)
    return fn


def _make_binary(builder):
    def fn(ast_self, rhs):
        return _eltwise_block(lambda l, r: builder(l.type, l, r), ast_self, rhs)
    return fn


for _name, _b in _UNARY_OPS.items():
    _fn = _make_unary(_b)
    _fn.__name__ = _name
    syntax(_name)(_fn)
    setattr(TensorBlock, _name, _fn)

for _name, _b in _BINARY_OPS.items():
    _fn = _make_binary(_b)
    _fn.__name__ = _name
    syntax(_name)(_fn)
    setattr(TensorBlock, _name, _fn)

# Apply syntax("!tensor") AFTER the method-style entries are populated.
TensorBlock = syntax("!tensor")(TensorBlock)


# --- Non-elementwise tile ops ---------------------------------------------
# These don't fit the simple _eltwise_block pattern (typecast changes
# element type, transpose has intra-tile semantics). Left in place so they
# can be migrated to their own bespoke builders later.


@syntax("tile_typecast")
def tile_typecast(input, result_type):
    return d2m.tile_typecast(result_type, input)


@syntax("tile_transpose")
def tile_transpose(input):
    return d2m.tile_transpose(input.type, input)


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


