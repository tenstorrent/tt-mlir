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

from ..layout_math import current_mesh
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


def remote_load(*args, mcast_start_index=None, mcast_shape=None, mcast_dims=None):
    if 2 <= len(args) <= 5 and isinstance(args[1], (list, tuple)):
        local_buffer, src, indices = None, args[0], args[1]
        positional_mcast = args[2:]
        mcast_values = [mcast_start_index, mcast_shape, mcast_dims]
        mcast_names = ["mcast_start_index", "mcast_shape", "mcast_dims"]
        for i, value in enumerate(positional_mcast):
            if mcast_values[i] is not None:
                raise TypeError(
                    f"remote_load got multiple values for '{mcast_names[i]}'"
                )
            mcast_values[i] = value
        mcast_start_index, mcast_shape, mcast_dims = mcast_values
    elif len(args) == 3:
        local_buffer, src, indices = args
    else:
        raise TypeError(
            "remote_load expects (src, indices[, mcast_start_index, "
            "mcast_shape, mcast_dims]) or (buffer, src, indices); "
            f"got {len(args)} positional arguments"
        )

    if not isinstance(src, SimTensor):
        raise TypeError(f"remote_load source must be a SimTensor, got {type(src)}")
    if local_buffer is not None and not isinstance(local_buffer, SimBlock):
        raise TypeError(
            f"remote_load buffer must be a SimBlock, got {type(local_buffer)}"
        )
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
    loaded = SimBlock.from_2d(sl)
    if local_buffer is None:
        return loaded
    if local_buffer.tile_grid != loaded.tile_grid:
        raise ValueError(
            f"remote_load shape mismatch: buffer is {local_buffer.tile_grid} tiles "
            f"but loaded block is {loaded.tile_grid} tiles"
        )
    local_buffer.tiles.copy_(loaded.tiles)
    return local_buffer


def remote_store(
    dst,
    indices,
    src,
    *,
    start_device=None,
    device_mcast_shape=None,
    semaphore=None,
    semaphore_indices=None,
):
    if not isinstance(dst, SimTensor):
        raise TypeError(f"remote_store dest must be a SimTensor, got {type(dst)}")
    if not isinstance(src, SimBlock):
        raise TypeError(f"remote_store value must be a SimBlock, got {type(src)}")
    if semaphore is not None and not isinstance(semaphore, Semaphore):
        raise TypeError(
            f"remote_store semaphore must be a Semaphore, got {type(semaphore)}"
        )
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
    if semaphore is not None:
        semaphore.inc(1)


def mesh_position(dim):
    """Return the coordinate of the simulator's single mesh device."""
    mesh = current_mesh()
    if mesh is None:
        raise RuntimeError("mesh_position() requires a preceding mesh() declaration")
    dim = int(dim)
    if not 0 <= dim < len(mesh["shape"]):
        raise IndexError(
            f"mesh_position dim {dim} out of bounds for mesh shape {mesh['shape']}"
        )
    return 0


def semaphore_wait(semaphore, value, reset=None):
    if not isinstance(semaphore, Semaphore):
        raise TypeError(f"semaphore_wait expected a Semaphore, got {type(semaphore)}")
    semaphore.wait(value, reset=reset)


def device_synchronize(
    semaphore,
    *,
    start_device=None,
    mcast_shape=None,
    num_receivers=0,
    core_indices=None,
):
    if not isinstance(semaphore, Semaphore):
        raise TypeError(
            f"device_synchronize expected a Semaphore, got {type(semaphore)}"
        )
    # Mesh devices and fabric traffic are not modeled. Kernel bodies execute
    # serially on one simulated device, so this synchronization is complete.


# --- elementwise helpers -----------------------------------------------------


def _common_reduced_axes(*blocks):
    # Mirror the device eltwise rule (api.py `_common_reduced_axes`): keep the
    # reduced-axes set only when every operand carries the exact same one,
    # otherwise clear it. The sim never branches on this field (reduction
    # results are eagerly broadcast to full block shape, so eltwise is correct
    # without it -- see SIMULATOR_SPEC.md §5.3); it is kept aligned with the
    # device purely for parity/diagnostics.
    axes = [b.reduced_axes for b in blocks]
    if axes and all(a == axes[0] for a in axes):
        return axes[0]
    return frozenset()


def _unary(fn, x):
    return SimBlock(fn(x.tiles), x.reduced_axes)


def _binary(fn, lhs, rhs):
    return SimBlock(fn(lhs.tiles, rhs.tiles), _common_reduced_axes(lhs, rhs))


def _predicate(pred, x):
    return SimBlock(pred(x.tiles).to(x.tiles.dtype), x.reduced_axes)


def _compare(cmp, lhs, rhs):
    # Comparisons write 1/0 into each tile lane in the operands' element type,
    # matching d2m.tile_eq / tile_gt / ... (which take the lhs tile type).
    return SimBlock(
        cmp(lhs.tiles, rhs.tiles).to(lhs.tiles.dtype),
        _common_reduced_axes(lhs, rhs),
    )


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

# Comparisons return 0/1 in the operands' tile dtype. Not wired to Python's
# `<` / `==` dunders -- api.py lowers `visit_Compare` to arith.cmpi for
# index-domain conditions, so the DSL exposes these by name only.
_COMPARE = {
    "eq": torch.eq,
    "ne": torch.ne,
    "gt": torch.gt,
    "ge": torch.ge,
    "lt": torch.lt,
    "le": torch.le,
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
    return SimBlock(out, _common_reduced_axes(cond, true_value, false_value))


def zeros(shape):
    """Kernel-body zero block of `shape` tiles, e.g. `zeros([m_tiles, n_tiles])`.

    The device op's tile type is always f32 regardless of the operand layouts,
    so the sim block is f32 too. Distinct from host-side `d2m.zeros(layout)`,
    which allocates a whole tensor. This is the loop-carried accumulator for an
    explicit K loop (`c = zeros([1, 1])` then `c += a @ b`); native `+=` on a
    SimBlock covers what the device lowers via `__matmul_acc__`.
    """
    if len(shape) != 2:
        raise NotImplementedError(
            f"sim zeros() supports rank-2 block shapes only, got {list(shape)}"
        )
    bm, bn = int(shape[0]), int(shape[1])
    return SimBlock(torch.zeros(bm, bn, TILE, TILE, dtype=torch.float32))


def empty(shape):
    """Kernel-body scratch block, deterministically zeroed in the simulator."""
    return zeros(shape)


def matmul(lhs, rhs, transpose_b=False):
    a = lhs.to_2d()
    b = rhs.to_2d()
    if transpose_b:
        b = b.transpose(0, 1)
    return SimBlock.from_2d(a @ b)


# --- reductions --------------------------------------------------------------


def _norm_axis(dim):
    # bool is an int subclass, so `int(True) == 1` would silently reduce columns.
    # The device rejects bool dims ("expected integer literal"); match that
    # rather than guessing an axis.
    if isinstance(dim, bool):
        raise TypeError(f"reduce dim must be an integer literal, got {dim!r}")
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


def _make_compare(cmp):
    return lambda lhs, rhs: _compare(cmp, lhs, rhs)


# Block ops reachable both as free functions and (most of them) as methods.
_BLOCK_OPS = {}
for _name, _fn in _UNARY.items():
    _BLOCK_OPS[_name] = _make_unary(_fn)
for _name, _fn in _BINARY.items():
    _BLOCK_OPS[_name] = _make_binary(_fn)
for _name, _pred in _PREDICATE.items():
    _BLOCK_OPS[_name] = _make_predicate(_pred)
for _name, _cmp in _COMPARE.items():
    _BLOCK_OPS[_name] = _make_compare(_cmp)
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
        "device_synchronize": device_synchronize,
        "empty": empty,
        "mesh_position": mesh_position,
        "remote_load": remote_load,
        "remote_store": remote_store,
        "semaphore_wait": semaphore_wait,
        "Semaphore": Semaphore,
        # Free function only -- there is no `!tensor.zeros` method form.
        "zeros": zeros,
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
