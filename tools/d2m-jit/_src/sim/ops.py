# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Torch implementations of every kernel-body op the AST compiler registers via
`@syntax`, plus the `SimBlock` dunders/methods and the free-function table
(`SIM_OPS`) the simulator injects into a kernel's globals.

Numerics are block-region level (see SIMULATOR_SPEC.md §7): pointwise ops apply
torch directly; matmul is a region matmul (accumulated in fp32); reductions are
keepdim (implicit broadcast falls out of torch broadcasting); only tile_bcast /
tile_transpose respect the 32x32 tile.
"""

from __future__ import annotations

import torch

from ..config import config
from ..errors import D2mJitError
from .block import SimBlock
from .tensor import torch_dtype

_TILE = 32


# --- numeric helpers --------------------------------------------------------


def _hp(data: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32 for compute when high-precision mode is on."""
    if config.sim_high_precision and data.is_floating_point():
        return data.float()
    return data


def _block_shape_of(region: torch.Tensor):
    h, w = region.shape
    return (h // _TILE, w // _TILE)


def _unary(fn):
    def op(x: SimBlock) -> SimBlock:
        r = fn(_hp(x.data)).to(x.data.dtype)
        return SimBlock(r, x.block_shape, x.reduced_axes)

    return op


def _binary(fn):
    def op(a: SimBlock, b: SimBlock) -> SimBlock:
        r = fn(_hp(a.data), _hp(b.data)).to(a.data.dtype)
        return SimBlock(r, _block_shape_of(r), a.reduced_axes & b.reduced_axes)

    return op


# --- op tables --------------------------------------------------------------
#
# Names mirror D2MCompiler._syntax exactly (see api.py). A coverage test asserts
# SIM_OPS keys are a superset of the supported syntax names.

_UNARY = {
    "recip": torch.reciprocal,
    "exp": torch.exp,
    "exp2": torch.exp2,
    "expm1": torch.expm1,
    "log": torch.log,
    "log1p": torch.log1p,
    "negative": torch.negative,
    "cos": torch.cos,
    "acos": torch.acos,
    "sin": torch.sin,
    "asin": torch.asin,
    "tan": torch.tan,
    "atan": torch.atan,
    "tanh": torch.tanh,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "square": lambda t: t * t,
    "sigmoid": torch.sigmoid,
    "hardsigmoid": torch.nn.functional.hardsigmoid,
    "silu": torch.nn.functional.silu,
    "softsign": lambda t: t / (1.0 + t.abs()),
    "selu": torch.nn.functional.selu,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "erf": torch.erf,
    "erfc": torch.erfc,
    "sign": torch.sign,
    "signbit": lambda t: torch.signbit(t),
    "ceil": torch.ceil,
    "floor": torch.floor,
    "frac": torch.frac,
    "trunc": torch.trunc,
    "abs": torch.abs,
    "bitwise_not": torch.bitwise_not,
    "logical_not": lambda t: t == 0,
    "eqz": lambda t: t == 0,
    "nez": lambda t: t != 0,
    "gtz": lambda t: t > 0,
    "gez": lambda t: t >= 0,
    "ltz": lambda t: t < 0,
    "lez": lambda t: t <= 0,
}

_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: torch.pow(a, b),
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "bitwise_and": torch.bitwise_and,
    "bitwise_or": torch.bitwise_or,
    "bitwise_xor": torch.bitwise_xor,
    "logical_left_shift": torch.bitwise_left_shift,
    "logical_right_shift": torch.bitwise_right_shift,
    "right_shift": torch.bitwise_right_shift,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
}

# Built into callables operating on SimBlock.
_UNARY_OPS = {name: _unary(fn) for name, fn in _UNARY.items()}
_BINARY_OPS = {name: _binary(fn) for name, fn in _BINARY.items()}


# --- structural / special ops ----------------------------------------------


def _bcast_kind(kind) -> str:
    if isinstance(kind, str):
        k = kind.lower()
        if k in ("row",):
            return "row"
        if k in ("col", "column"):
            return "col"
        if k in ("2d", "scalar"):
            return "scalar"
        if k in ("none", "no_bcast"):
            return "none"
    # d2m.TileBcastType enum or int -> compare by str form.
    s = str(kind).lower()
    for cand in ("row", "col", "scalar"):
        if cand in s:
            return cand
    raise ValueError(f"unsupported tile broadcast type: {kind!r}")


def tile_bcast(x: SimBlock, bcast_type) -> SimBlock:
    kind = _bcast_kind(bcast_type)
    t = x.as_tiles()  # [Bm, Bn, 32, 32]
    if kind == "none":
        return SimBlock(x.data.clone(), x.block_shape)
    if kind == "row":
        t2 = t[:, :, 0:1, :].expand_as(t)
    elif kind == "col":
        t2 = t[:, :, :, 0:1].expand_as(t)
    else:  # scalar / 2d
        t2 = t[:, :, 0:1, 0:1].expand_as(t)
    return SimBlock.from_tiles(t2.contiguous(), x.block_shape)


def tile_bcast_row(x: SimBlock) -> SimBlock:
    return tile_bcast(x, "row")


def tile_bcast_col(x: SimBlock) -> SimBlock:
    return tile_bcast(x, "col")


def tile_bcast_2d(x: SimBlock) -> SimBlock:
    return tile_bcast(x, "scalar")


def tile_transpose(x: SimBlock) -> SimBlock:
    t = x.as_tiles().transpose(-1, -2).contiguous()
    return SimBlock.from_tiles(t, x.block_shape, x.reduced_axes)


def typecast(x: SimBlock, dtype) -> SimBlock:
    return SimBlock(x.data.to(torch_dtype(dtype)), x.block_shape, x.reduced_axes)


def clamp_scalar(x: SimBlock, lo, hi) -> SimBlock:
    return SimBlock(x.data.clamp(float(lo), float(hi)), x.block_shape, x.reduced_axes)


def where(cond: SimBlock, true_value: SimBlock, false_value: SimBlock) -> SimBlock:
    r = torch.where(cond.data != 0, true_value.data, false_value.data)
    return SimBlock(
        r,
        _block_shape_of(r),
        true_value.reduced_axes & false_value.reduced_axes,
    )


def store(dst: SimBlock, src: SimBlock) -> SimBlock:
    """Write `src` into `dst` in place and return `dst` (d2m.store: same-typed
    operands, result aliases the destination)."""
    dst.data = src.data.to(dst.data.dtype).clone()
    dst.reduced_axes = src.reduced_axes
    return dst


def zeros(shape) -> SimBlock:
    m, n = int(shape[0]), int(shape[1])
    return SimBlock(torch.zeros(m * _TILE, n * _TILE, dtype=torch.float32), (m, n))


def matmul(lhs: SimBlock, rhs: SimBlock, transpose_b=False) -> SimBlock:
    a = lhs.data.float()
    b = rhs.data.float()
    if transpose_b:
        b = b.transpose(-1, -2)
    r = (a @ b).to(lhs.data.dtype)
    return SimBlock(r, _block_shape_of(r))


def _norm_reduce_dim(dim) -> int:
    # Raise D2mJitError (not ValueError) so the sim matches the compiler path's
    # error type for the same misuse; the messages carry the same substrings.
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise D2mJitError("expected integer literal for reduce dim")
    if dim in (0, -2):
        return 0
    if dim in (1, -1):
        return 1
    raise D2mJitError(f"reduce dim must be 0/1 or -2/-1, got {dim}")


def _reduce(reduce_fn, x: SimBlock, dim) -> SimBlock:
    axis = _norm_reduce_dim(dim)
    red = reduce_fn(_hp(x.data), axis).to(x.data.dtype)
    bs = list(x.block_shape)
    bs[axis] = 1
    return SimBlock(red, bs, x.reduced_axes | {axis})


def reduce_sum(x: SimBlock, dim) -> SimBlock:
    return _reduce(lambda d, a: torch.sum(d, dim=a, keepdim=True), x, dim)


def reduce_max(x: SimBlock, dim) -> SimBlock:
    return _reduce(lambda d, a: torch.amax(d, dim=a, keepdim=True), x, dim)


def reduce_mean(x: SimBlock, dim) -> SimBlock:
    return _reduce(lambda d, a: torch.mean(d, dim=a, keepdim=True), x, dim)


# --- assemble the free-function table --------------------------------------

SIM_OPS = {}
SIM_OPS.update(_UNARY_OPS)
SIM_OPS.update(_BINARY_OPS)
SIM_OPS.update(
    {
        "clamp_scalar": clamp_scalar,
        "typecast": typecast,
        "tile_transpose": tile_transpose,
        "tile_bcast": tile_bcast,
        "tile_bcast_row": tile_bcast_row,
        "tile_bcast_col": tile_bcast_col,
        "tile_bcast_2d": tile_bcast_2d,
        "where": where,
        "zeros": zeros,
        "matmul": matmul,
        "reduce_sum": reduce_sum,
        "reduce_max": reduce_max,
        "reduce_mean": reduce_mean,
    }
)


# --- attach dunders + method forms to SimBlock ------------------------------
#
# Done here (not in block.py) so the whole op set is defined in one place. Only
# the ops that have method forms in api.TensorBlock are attached; clamp_scalar /
# typecast / tile_bcast keep free-function-only forms, matching the DSL.

_DUNDERS = {
    "__add__": "add",
    "__sub__": "sub",
    "__mul__": "mul",
    "__truediv__": "div",
    "__matmul__": "__matmul__",
    "__neg__": "negative",
    "__invert__": "bitwise_not",
}


def _install_methods():
    for name, fn in _UNARY_OPS.items():
        setattr(SimBlock, name, (lambda fn: lambda self: fn(self))(fn))
    for name, fn in _BINARY_OPS.items():
        setattr(SimBlock, name, (lambda fn: lambda self, other: fn(self, other))(fn))

    setattr(SimBlock, "tile_bcast_row", lambda self: tile_bcast_row(self))
    setattr(SimBlock, "tile_bcast_col", lambda self: tile_bcast_col(self))
    setattr(SimBlock, "tile_bcast_2d", lambda self: tile_bcast_2d(self))
    setattr(SimBlock, "tile_transpose", lambda self: tile_transpose(self))
    setattr(SimBlock, "where", lambda self, t, f: where(self, t, f))
    setattr(SimBlock, "store", lambda self, rhs: store(self, rhs))
    setattr(
        SimBlock,
        "matmul",
        lambda self, other, transpose_b=False: matmul(
            self, other, transpose_b=transpose_b
        ),
    )
    setattr(SimBlock, "reduce_sum", lambda self, dim: reduce_sum(self, dim))
    setattr(SimBlock, "reduce_max", lambda self, dim: reduce_max(self, dim))
    setattr(SimBlock, "reduce_mean", lambda self, dim: reduce_mean(self, dim))

    # Operator dunders.
    setattr(SimBlock, "__add__", lambda self, other: _BINARY_OPS["add"](self, other))
    setattr(SimBlock, "__sub__", lambda self, other: _BINARY_OPS["sub"](self, other))
    setattr(SimBlock, "__mul__", lambda self, other: _BINARY_OPS["mul"](self, other))
    setattr(
        SimBlock, "__truediv__", lambda self, other: _BINARY_OPS["div"](self, other)
    )
    setattr(SimBlock, "__matmul__", lambda self, other: matmul(self, other))
    setattr(SimBlock, "__neg__", lambda self: _UNARY_OPS["negative"](self))
    setattr(SimBlock, "__invert__", lambda self: _UNARY_OPS["bitwise_not"](self))


_install_methods()
