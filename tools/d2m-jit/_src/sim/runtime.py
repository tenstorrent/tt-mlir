# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-core execution context and the shard-DMA builtins (`core_index`,
`remote_load`, `remote_store`).

A kernel runs once per grid core; `run_kernel` iterates the grid and binds the
active `CoreContext` so the injected builtins can find the current `(cy, cx)`.
Block indices are **global** grid-dimension indices (per the D2M op contract:
"indices correspond to the grid dimensions only"), so `remote_load`/`remote_store`
address the target's tile grid directly regardless of which core is running.
"""

from __future__ import annotations

import threading

from .block import SimBlock

_local = threading.local()


class CoreContext:
    __slots__ = ("cy", "cx")

    def __init__(self, cy, cx):
        self.cy = cy
        self.cx = cx


def _core() -> CoreContext:
    ctx = getattr(_local, "core", None)
    if ctx is None:
        raise RuntimeError("no active core: kernel-body op called outside run_kernel")
    return ctx


# --- kernel-body builtins ---------------------------------------------------


def core_index(index):
    c = _core()
    return c.cy if index == 0 else c.cx


def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
):
    """Load a shard from `src` at global grid index `indices`.

    Multicast kwargs describe NOC routing only; the data loaded is the block at
    `indices` either way, so the simulator ignores them (SIMULATOR_SPEC §5.3).
    """
    st = src._sim_resolve()
    idx = [int(i) for i in indices]
    region = st.read_block(idx)
    return SimBlock(region, tuple(st.layout.block_shape))


def remote_store(dst, indices, src: SimBlock):
    st = dst._sim_resolve()
    idx = [int(i) for i in indices]
    st.write_block(idx, src.data)


# --- driver -----------------------------------------------------------------


def run_kernel(sim_fn, lazy_args, scalar_args, grid):
    """Execute `sim_fn` once per grid core with the sim builtins bound.

    Outputs are `SimLazyTensor`s whose backing `SimTensor` is mutated in place
    by `remote_store`, so the caller reads results straight off them afterward.
    """
    gy, gx = grid
    prev = getattr(_local, "core", None)
    try:
        for cy in range(gy):
            for cx in range(gx):
                _local.core = CoreContext(cy, cx)
                sim_fn(*lazy_args, *scalar_args)
    finally:
        _local.core = prev
