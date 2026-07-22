# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""torch backings for every in-kernel `@syntax` op (see api.py).

Two registries are exported:

- `SIM_OPS`     -- the namespace injected into a running kernel body
                   (`core_index`, `remote_load`, `remote_store`, and all the
                   free-function block ops).
- `SIM_METHODS` -- the block ops reachable as `SimBlock` method forms
                   (`x.exp()`, `x.reduce_max(0)`, ...).

All math runs in the block's torch dtype (exact mode). Device-quirk numerics
(fp19 fills, reduced-precision accumulate) are out of scope for v1.
"""

import threading

import torch
import torch.nn.functional as F

from ..tensor_layout import _to_data_type
from .tensors import SimBlock, SimTensor, block_extent, TILE

# --- current-core thread-local (SPMD) ---------------------------------------

_state = threading.local()


def _set_current_core(core):
    _state.core = core


def core_index(index):
    core = getattr(_state, "core", None)
    if core is None:
        raise RuntimeError("core_index() called outside a kernel SPMD loop")
    return int(core[int(index)])


# --- synchronization (async / semaphores) -----------------------------------


class Semaphore:
    """A NOC synchronization semaphore.

    On device, semaphores order the async data-movement / compute threads of a
    multi-thread kernel. The functional sim runs a kernel body straight through
    in program order on a single thread, so waits are always already satisfied
    and set/inc/wait are no-ops kept only so async kernels referencing them run.
    `await sem` resolves immediately. Semaphores are ordering-only and do not
    affect numerics (see SIMULATOR_SPEC.md §5.5 / §13 non-goals).
    """

    __slots__ = ("_value",)

    def __init__(self, value=0):
        self._value = int(value)

    def set(self, value, core=None, mcast=None):
        self._value = int(value)

    def inc(self, value, core=None, mcast=None):
        self._value += int(value)

    def wait(self, value, reset=None):
        # Synchronous sim: the awaited condition already holds. Honor an
        # explicit reset so a subsequent wait sees the reset value.
        if reset is not None:
            self._value = int(reset)

    def __await__(self):
        yield from ()
        return self


# --- movement ----------------------------------------------------------------


def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
):
    if not isinstance(src, SimTensor):
        raise TypeError(f"remote_load source must be a SimTensor, got {type(src)}")
    if len(indices) != 2:
        raise NotImplementedError("sim remote_load supports rank-2 indices only")
    i, j = int(indices[0]), int(indices[1])
    em, en = block_extent(src.layout)
    rows, cols = src.buffer.shape
    if (i + 1) * em > rows or (j + 1) * en > cols:
        raise IndexError(
            f"remote_load block [{i}, {j}] out of bounds for buffer {(rows, cols)}"
        )
    sl = src.buffer[i * em : (i + 1) * em, j * en : (j + 1) * en]
    return SimBlock.from_2d(sl)


def remote_store(dst, indices, src):
    if not isinstance(dst, SimTensor):
        raise TypeError(f"remote_store dest must be a SimTensor, got {type(dst)}")
    if not isinstance(src, SimBlock):
        raise TypeError(f"remote_store value must be a SimBlock, got {type(src)}")
    if len(indices) != 2:
        raise NotImplementedError("sim remote_store supports rank-2 indices only")
    i, j = int(indices[0]), int(indices[1])
    em, en = block_extent(dst.layout)
    rows, cols = dst.buffer.shape
    if (i + 1) * em > rows or (j + 1) * en > cols:
        raise IndexError(
            f"remote_store block [{i}, {j}] out of bounds for buffer {(rows, cols)}"
        )
    patch = src.to_2d()
    if tuple(patch.shape) != (em, en):
        raise ValueError(
            f"remote_store shape mismatch: block is {tuple(patch.shape)} but "
            f"destination block is {(em, en)}"
        )
    dst.buffer[i * em : (i + 1) * em, j * en : (j + 1) * en] = patch.to(
        dst.buffer.dtype
    )


# --- elementwise helpers -----------------------------------------------------


def _unary(fn, x):
    return SimBlock(fn(x.tiles), x.reduced_axes)


def _binary(fn, lhs, rhs):
    reduced = lhs.reduced_axes & rhs.reduced_axes
    return SimBlock(fn(lhs.tiles, rhs.tiles), reduced)


def _predicate(pred, x):
    return SimBlock(pred(x.tiles).to(x.tiles.dtype), x.reduced_axes)


# Plain unary ops backed directly by a torch callable.
_UNARY = {
    "recip": torch.reciprocal,
    "exp": torch.exp,
    "exp2": torch.exp2,
    "expm1": torch.expm1,
    "log": torch.log,
    "log1p": torch.log1p,
    "negative": torch.neg,
    "cos": torch.cos,
    "acos": torch.acos,
    "sin": torch.sin,
    "asin": torch.asin,
    "tan": torch.tan,
    "atan": torch.atan,
    "tanh": torch.tanh,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "square": torch.square,
    "sigmoid": torch.sigmoid,
    "hardsigmoid": F.hardsigmoid,
    "silu": F.silu,
    "selu": F.selu,
    "softsign": F.softsign,
    "relu": torch.relu,
    "gelu": F.gelu,
    "erf": torch.erf,
    "erfc": torch.erfc,
    "sign": torch.sign,
    "ceil": torch.ceil,
    "floor": torch.floor,
    "frac": torch.frac,
    "trunc": torch.trunc,
    "abs": torch.abs,
    "bitwise_not": torch.bitwise_not,
}

_BINARY = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "pow": torch.pow,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "bitwise_and": torch.bitwise_and,
    "bitwise_or": torch.bitwise_or,
    "bitwise_xor": torch.bitwise_xor,
    "logical_left_shift": torch.bitwise_left_shift,
    "logical_right_shift": torch.bitwise_right_shift,
    "right_shift": torch.bitwise_right_shift,
}

# Predicates return 0.0/1.0 in the tile dtype.
_PREDICATE = {
    "signbit": torch.signbit,
    "logical_not": lambda t: t == 0,
    "eqz": lambda t: t == 0,
    "nez": lambda t: t != 0,
    "gtz": lambda t: t > 0,
    "gez": lambda t: t >= 0,
    "ltz": lambda t: t < 0,
    "lez": lambda t: t <= 0,
}


# --- bespoke ops -------------------------------------------------------------


def clamp_scalar(x, min, max):
    return SimBlock(x.tiles.clamp(float(min), float(max)), x.reduced_axes)


def typecast(x, dtype):
    name = _to_data_type(dtype).name
    from .tensors import _TORCH_BY_NAME

    return SimBlock(x.tiles.to(_TORCH_BY_NAME[name]), x.reduced_axes)


def tile_transpose(x):
    # Per-tile (32x32) transpose; the block tile-grid is unchanged.
    return SimBlock(x.tiles.transpose(2, 3).contiguous(), x.reduced_axes)


def _bcast_kind(bcast_type):
    if hasattr(bcast_type, "name"):
        bcast_type = bcast_type.name
    key = str(bcast_type).lower()
    if key in {"row"}:
        return "row"
    if key in {"col", "column"}:
        return "col"
    if key in {"2d", "scalar"}:
        return "2d"
    raise ValueError(f"unknown tile broadcast type {bcast_type!r}")


def tile_bcast(x, bcast_type):
    kind = _bcast_kind(bcast_type)
    t = x.tiles
    bm, bn, th, tw = t.shape
    if kind == "row":
        out = t[:, :, 0:1, :].expand(bm, bn, th, tw)
    elif kind == "col":
        out = t[:, :, :, 0:1].expand(bm, bn, th, tw)
    else:  # 2d / scalar
        out = t[:, :, 0:1, 0:1].expand(bm, bn, th, tw)
    return SimBlock(out.contiguous())


def tile_bcast_row(x):
    return tile_bcast(x, "row")


def tile_bcast_col(x):
    return tile_bcast(x, "col")


def tile_bcast_2d(x):
    return tile_bcast(x, "2d")


def where(cond, true_value, false_value):
    out = torch.where(cond.tiles != 0, true_value.tiles, false_value.tiles)
    return SimBlock(out)


def matmul(lhs, rhs, transpose_b=False):
    a = lhs.to_2d()
    b = rhs.to_2d()
    if transpose_b:
        b = b.transpose(0, 1)
    return SimBlock.from_2d(a @ b)


# --- reductions --------------------------------------------------------------


def _norm_axis(dim):
    d = int(dim)
    if d in (0, -2):
        return 0
    if d in (1, -1):
        return 1
    raise ValueError(f"reduce dim must be 0/1 or -2/-1, got {dim}")


def _reduce(kind, x, dim):
    axis = _norm_axis(dim)
    t = x.tiles
    dims = (0, 2) if axis == 0 else (1, 3)
    if kind == "sum":
        r = t.sum(dim=dims, keepdim=True)
    elif kind == "max":
        r = torch.amax(t, dim=dims, keepdim=True)
    elif kind == "mean":
        r = t.mean(dim=dims, keepdim=True)
    else:
        raise ValueError(kind)
    # Reduced tile-axis collapses to one tile; the value is broadcast across
    # the within-tile direction so a reduction_layout readback (which slices
    # row/col 0) and implicit eltwise broadcast both see the right number.
    shape = list(t.shape)
    if axis == 0:
        shape[0] = 1
        shape[2] = TILE
    else:
        shape[1] = 1
        shape[3] = TILE
    return SimBlock(r.expand(shape).contiguous(), reduced_axes={axis})


def reduce_sum(x, dim):
    return _reduce("sum", x, dim)


def reduce_max(x, dim):
    return _reduce("max", x, dim)


def reduce_mean(x, dim):
    return _reduce("mean", x, dim)


# --- registries --------------------------------------------------------------


def _make_unary(fn):
    return lambda x: _unary(fn, x)


def _make_binary(fn):
    return lambda lhs, rhs: _binary(fn, lhs, rhs)


def _make_predicate(pred):
    return lambda x: _predicate(pred, x)


# Block ops reachable both as free functions and (most of them) as methods.
_BLOCK_OPS = {}
for _name, _fn in _UNARY.items():
    _BLOCK_OPS[_name] = _make_unary(_fn)
for _name, _fn in _BINARY.items():
    _BLOCK_OPS[_name] = _make_binary(_fn)
for _name, _pred in _PREDICATE.items():
    _BLOCK_OPS[_name] = _make_predicate(_pred)
_BLOCK_OPS.update(
    {
        "clamp_scalar": clamp_scalar,
        "typecast": typecast,
        "tile_transpose": tile_transpose,
        "tile_bcast": tile_bcast,
        "tile_bcast_row": tile_bcast_row,
        "tile_bcast_col": tile_bcast_col,
        "tile_bcast_2d": tile_bcast_2d,
        "where": where,
        "matmul": matmul,
        "reduce_sum": reduce_sum,
        "reduce_max": reduce_max,
        "reduce_mean": reduce_mean,
    }
)

# Method-form dispatch for SimBlock.__getattr__.
SIM_METHODS = dict(_BLOCK_OPS)

# Namespace injected into a running kernel body.
SIM_OPS = dict(_BLOCK_OPS)
SIM_OPS.update(
    {
        "core_index": core_index,
        "remote_load": remote_load,
        "remote_store": remote_store,
        "Semaphore": Semaphore,
    }
)

# Bind the free functions as real module attributes too (so `from .ops import
# add` works for the SimBlock dunders).
add = _BLOCK_OPS["add"]
sub = _BLOCK_OPS["sub"]
mul = _BLOCK_OPS["mul"]
div = _BLOCK_OPS["div"]
negative = _BLOCK_OPS["negative"]
bitwise_not = _BLOCK_OPS["bitwise_not"]
