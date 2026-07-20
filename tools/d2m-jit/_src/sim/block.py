# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SimBlock: a loaded shard the kernel body manipulates.

Mirrors `api.TensorBlock` -- same operator dunders and method-form ops -- but
its value is a torch region tensor of shape `(Bm*32, Bn*32)` (tiled) rather than
an MLIR tensor-of-tiles. The dunders/methods are attached by `ops.py` (imported
for side effects) so the op set lives in one place.
"""

from __future__ import annotations

import torch

_TILE = 32


class SimBlock:
    """A block of tiles as a torch region + tile bookkeeping.

    `reduced_axes` tracks logical axes already reduced (0 = rows, 1 = cols),
    matching `api._REDUCED_AXES_ATTR`; it drives implicit-broadcast in eltwise
    (a reduced operand keeps a size-1 axis and torch broadcasts it back).
    """

    __slots__ = ("data", "block_shape", "reduced_axes")

    def __init__(self, data: torch.Tensor, block_shape, reduced_axes=frozenset()):
        self.data = data
        self.block_shape = tuple(block_shape)
        self.reduced_axes = frozenset(reduced_axes)

    # --- tile <-> region reshapes (for per-tile structural ops) ------------

    def as_tiles(self) -> torch.Tensor:
        """View the region as `[Bm, Bn, 32, 32]`."""
        h, w = self.data.shape
        bm, bn = h // _TILE, w // _TILE
        return self.data.reshape(bm, _TILE, bn, _TILE).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def from_tiles(
        tiles: torch.Tensor, block_shape, reduced_axes=frozenset()
    ) -> "SimBlock":
        bm, bn = tiles.shape[0], tiles.shape[1]
        region = tiles.permute(0, 2, 1, 3).reshape(bm * _TILE, bn * _TILE)
        return SimBlock(region, block_shape, reduced_axes)

    def __repr__(self):
        return (
            f"SimBlock(shape={tuple(self.data.shape)}, block_shape={self.block_shape}, "
            f"reduced_axes={set(self.reduced_axes)})"
        )
