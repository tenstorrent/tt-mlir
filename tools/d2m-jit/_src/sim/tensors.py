# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Value types for the d2m-jit simulator.

`SimTensor`  -- host handle, the sim analog of `LazyTensor`. Wraps a single
                tile-padded torch buffer plus its `Layout`.
`SimBlock`   -- the in-kernel `!tensor` tile-block value that `remote_load`
                returns and the block-level ops consume. A 2-D block of
                `bm x bn` tiles of 32x32, stored as a torch tensor of shape
                `(bm, bn, 32, 32)` so torch broadcasting handles both the
                tile-axis and within-tile broadcast cases for free.
"""

import torch

from ..tensor_layout import Layout

TILE = 32


# --- ttcore.DataType -> torch dtype -----------------------------------------
#
# `Layout.dtype` is a `ttcore.DataType`. Compare by `.name` (the enum is bound
# by multiple nanobind modules, so `==` / `in` across modules is unreliable --
# see the note in api.py).
_TORCH_BY_NAME = {
    "Float32": torch.float32,
    "Float16": torch.float16,
    "BFloat16": torch.bfloat16,
}


def torch_dtype(layout: Layout):
    name = layout.dtype.name
    if name not in _TORCH_BY_NAME:
        raise ValueError(f"no torch dtype for ttcore.DataType {name}")
    return _TORCH_BY_NAME[name]


def _ceil_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def tile_padded_shape(layout: Layout):
    """Device-buffer shape: logical shape rounded up to the tile grid."""
    shape = layout.logical_shape
    if len(shape) != 2:
        raise NotImplementedError(
            f"sim supports rank-2 tensors only, got shape {shape}"
        )
    if layout.tiled:
        return [_ceil_tile(shape[0]), _ceil_tile(shape[1])]
    return list(shape)


def block_extent(layout: Layout):
    """Per-axis element extent of one (blocked-grid) block."""
    tile = TILE if layout.tiled else 1
    return [bs * tile for bs in layout.block_shape]


# --- SimBlock ----------------------------------------------------------------


class SimBlock:
    """In-kernel tile-block value. `.tiles` is a torch tensor of shape
    `(bm, bn, 32, 32)`."""

    __slots__ = ("tiles", "reduced_axes")

    def __init__(self, tiles, reduced_axes=frozenset()):
        self.tiles = tiles
        self.reduced_axes = frozenset(reduced_axes)

    # --- conversions ----
    @property
    def tile_grid(self):
        return (self.tiles.shape[0], self.tiles.shape[1])

    def to_2d(self):
        bm, bn, th, tw = self.tiles.shape
        return self.tiles.permute(0, 2, 1, 3).contiguous().reshape(bm * th, bn * tw)

    @staticmethod
    def from_2d(t, reduced_axes=frozenset()):
        rows, cols = t.shape
        if rows % TILE != 0 or cols % TILE != 0:
            raise ValueError(f"block shape {(rows, cols)} is not tile-aligned (32x32)")
        bm, bn = rows // TILE, cols // TILE
        tiles = (
            t.contiguous().reshape(bm, TILE, bn, TILE).permute(0, 2, 1, 3).contiguous()
        )
        return SimBlock(tiles, reduced_axes)

    # --- operator dunders (mirror api.py TensorBlock) ----
    def __add__(self, rhs):
        from . import ops

        return ops.add(self, rhs)

    def __sub__(self, rhs):
        from . import ops

        return ops.sub(self, rhs)

    def __mul__(self, rhs):
        from . import ops

        return ops.mul(self, rhs)

    def __truediv__(self, rhs):
        from . import ops

        return ops.div(self, rhs)

    def __neg__(self):
        from . import ops

        return ops.negative(self)

    def __invert__(self):
        from . import ops

        return ops.bitwise_not(self)

    def __matmul__(self, rhs):
        from . import ops

        return ops.matmul(self, rhs)

    def __await__(self):
        # `await block` in an async kernel marks "wait for this async DMA to
        # complete". Every device op in the functional sim is already
        # synchronous, so the await resolves immediately to the block itself
        # (yields nothing, so it never suspends the driving coroutine).
        yield from ()
        return self

    # --- method-style ops: dispatch generically to the op registry ----
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        from . import ops

        fn = ops.SIM_METHODS.get(name)
        if fn is None:
            raise AttributeError(f"SimBlock has no method '{name}'")
        return lambda *args, **kwargs: fn(self, *args, **kwargs)

    def __repr__(self):
        return f"SimBlock(tile_grid={self.tile_grid}, dtype={self.tiles.dtype})"


# --- SimTensor ---------------------------------------------------------------


class SimTensor:
    """Host handle wrapping a tile-padded torch buffer + its Layout."""

    __slots__ = ("layout", "buffer", "is_view")

    def __init__(self, layout: Layout, buffer, is_view: bool = False):
        self.layout = layout
        self.buffer = buffer
        self.is_view = is_view

    def to_logical(self):
        """Slice the device buffer back to the logical shape (a clone)."""
        rows, cols = self.layout.logical_shape
        return self.buffer[:rows, :cols].clone()

    def to_host(self):
        from .host import to_host

        return to_host(self)[0]

    def __await__(self):
        # See SimBlock.__await__: awaiting a device tensor is synchronous in the
        # functional sim and resolves immediately to the tensor itself.
        yield from ()
        return self

    def __repr__(self):
        return (
            f"SimTensor(shape={self.layout.logical_shape}, "
            f"dtype={self.buffer.dtype}, is_view={self.is_view})"
        )
