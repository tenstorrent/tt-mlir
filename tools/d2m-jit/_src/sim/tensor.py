# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SimTensor: a device tensor in the torch-backed simulator.

Backs a `SimLazyTensor`. Stores the data in **physical, tile-aligned** form
(each spatial dim padded up to a multiple of the tile size) so that:

  - `remote_load([i, j])` is a plain slice of the physical buffer, because a
    block of tiles is a contiguous logical sub-region (block `[i, j]` owns tile
    rows `i*Bm : (i+1)*Bm` and tile cols `j*Bn : (j+1)*Bn`), and
  - reduction outputs whose logical extent is < 32 along an axis still have a
    full 32-wide tile to live in; `to_logical()` crops back to the logical
    shape on the way out.

See SIMULATOR_SPEC.md §4 for the rationale.
"""

from __future__ import annotations

import torch

from ..tensor_layout import _to_data_type

_NAME_TO_TORCH = {
    "Float32": torch.float32,
    "Float16": torch.float16,
    "BFloat16": torch.bfloat16,
}


def torch_dtype(dtype) -> torch.dtype:
    """Map a d2m dtype (ttcore.DataType / string / torch.dtype) to a torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    return _NAME_TO_TORCH[_to_data_type(dtype).name]


def _tile(layout) -> int:
    return 32 if layout.tiled else 1


def _physical_shape(layout):
    """Logical shape rounded up to a whole number of tiles per axis."""
    t = _tile(layout)
    return [((d + t - 1) // t) * t for d in layout.logical_shape]


class SimTensor:
    """Mutable physical buffer + its Layout. Kernels mutate outputs in place."""

    __slots__ = ("layout", "physical")

    def __init__(self, layout, physical: torch.Tensor):
        self.layout = layout
        self.physical = physical

    # --- construction / extraction ----------------------------------------

    @classmethod
    def from_torch(cls, layout, t: torch.Tensor) -> "SimTensor":
        """Pad a logical torch tensor into a tile-aligned physical buffer."""
        t = t.to(torch_dtype(layout.dtype))
        phys_shape = _physical_shape(layout)
        if list(t.shape) == phys_shape:
            return cls(layout, t.clone())
        buf = torch.zeros(phys_shape, dtype=t.dtype)
        slices = tuple(slice(0, d) for d in layout.logical_shape)
        buf[slices] = t
        return cls(layout, buf)

    @classmethod
    def empty(cls, layout) -> "SimTensor":
        # Zero-filled rather than uninitialised: sim has no perf notion and
        # zeros keep padding lanes clean for reduction-shaped outputs.
        return cls(
            layout,
            torch.zeros(_physical_shape(layout), dtype=torch_dtype(layout.dtype)),
        )

    def to_logical(self) -> torch.Tensor:
        """Crop the physical buffer back to the layout's logical shape."""
        slices = tuple(slice(0, d) for d in self.layout.logical_shape)
        return self.physical[slices].clone()

    # --- block access (grid-dimension indexed) -----------------------------

    def _block_slice(self, indices):
        t = _tile(self.layout)
        block = self.layout.block_shape
        out = []
        for axis, idx in enumerate(indices):
            span = block[axis] * t
            start = idx * span
            out.append(slice(start, start + span))
        # Any trailing axes not addressed by `indices` are taken whole.
        for axis in range(len(indices), self.physical.dim()):
            out.append(slice(None))
        return tuple(out)

    def read_block(self, indices) -> torch.Tensor:
        return self.physical[self._block_slice(indices)].clone()

    def write_block(self, indices, region: torch.Tensor):
        sl = self._block_slice(indices)
        target = self.physical[sl]
        region = region.to(self.physical.dtype)
        if tuple(region.shape) != tuple(target.shape):
            # Reduction outputs arrive keepdim (a size-1 axis); broadcast the
            # value across the destination block before writing.
            region = region.expand(target.shape)
        self.physical[sl] = region
