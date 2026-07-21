# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-core execution context and the shard-DMA builtins (`core_index`,
`remote_load`, `remote_store`).

A kernel runs once per grid core; `run_kernel` iterates the grid and binds the
active `CoreContext` so the injected builtins can find the current `(cy, cx)`.

Store-index frame (SIMULATOR_CROSS_CORE_PROPOSAL.md). `remote_store` indices are
grid-dimension indices, but their frame depends on the kernel shape (confirmed
against device, Phase 0):

  - **All-parallel kernels** write a **global** index -- the body adds the core
    offset itself (e.g. `core_index(0)*m_blocks + m`). Store target = index.
  - **Multicast kernels** write a **core-relative** index along the grid dims
    the multicast collapses: the loads carry `mcast_shape`/`mcast_dims`, and for
    a collapsed dim the device supplies the core's own grid coordinate. Store
    target[d] = core_coord[d] (one block per core) + index[d].

The simulator detects collapsed dims from the `remote_load` multicast arguments
seen during the current core-run and resolves the store accordingly.
"""

from __future__ import annotations

import inspect
import threading

from .block import SimBlock

_local = threading.local()

_TILE = 32


class CoreContext:
    __slots__ = ("cy", "cx", "mcast_dims")

    def __init__(self, cy, cx):
        self.cy = cy
        self.cx = cx
        # Grid dims collapsed by multicast during this core-run (populated by
        # remote_load, read by remote_store).
        self.mcast_dims = set()

    def coord(self, dim):
        return self.cy if dim == 0 else self.cx


def _core() -> CoreContext:
    ctx = getattr(_local, "core", None)
    if ctx is None:
        raise RuntimeError("no active core: kernel-body op called outside run_kernel")
    return ctx


# --- kernel-body builtins ---------------------------------------------------


def core_index(index):
    c = _core()
    return c.cy if index == 0 else c.cx


def _record_mcast_dims(ctx, mcast_shape, mcast_dims):
    """Note which grid dims a load multicasts over, so a later store in the same
    core-run resolves core-relative along them (see module docstring). A dim is
    collapsed if its multicast span is > 1, or it is named in `mcast_dims`."""
    if mcast_shape is not None:
        for d, span in enumerate(mcast_shape):
            if int(span) > 1:
                ctx.mcast_dims.add(d)
    if mcast_dims is not None:
        for d in mcast_dims:
            ctx.mcast_dims.add(int(d))


def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
):
    """Load a shard from `src` at global grid index `indices`.

    The data loaded is the block at `indices` regardless of multicast (multicast
    is NOC routing only), but the multicast arguments are recorded so a store in
    the same core-run knows which grid dims are core-relative.
    """
    ctx = _core()
    _record_mcast_dims(ctx, mcast_shape, mcast_dims)
    st = src._sim_resolve()
    idx = [int(i) for i in indices]
    region = st.read_block(idx)
    return SimBlock(region, tuple(st.layout.block_shape))


def _blocks_per_core(layout, dim):
    """Output blocks the current core owns along `dim` (device-shape block count
    divided by the grid extent)."""
    tile = _TILE if layout.tiled else 1
    total_tiles = layout.logical_shape[dim] // tile
    total_blocks = total_tiles // layout.block_shape[dim]
    grid = layout.grid_shape[dim]
    return total_blocks // grid


def remote_store(dst, indices, src: SimBlock):
    st = dst._sim_resolve()
    ctx = _core()
    idx = [int(i) for i in indices]
    if ctx.mcast_dims:
        idx = _resolve_core_relative(st.layout, ctx, idx)
    st.write_block(idx, src.data)


def _resolve_core_relative(layout, ctx, idx):
    """Map a core-relative store index to a global grid index along the dims a
    multicast collapsed. Only one output block per core along a collapsed dim is
    supported; the general (block_factors > 1) case needs the generic op's
    indexing maps (Option A, SIMULATOR_CROSS_CORE_PROPOSAL.md §3)."""
    resolved = list(idx)
    for d in ctx.mcast_dims:
        if d >= len(resolved):
            continue
        per_core = _blocks_per_core(layout, d)
        if per_core != 1:
            raise NotImplementedError(
                "simulator supports core-relative (multicast) stores only with "
                f"one output block per core; grid dim {d} has {per_core} blocks "
                "per core. Modeling multi-block indexing maps is Option A in "
                "SIMULATOR_CROSS_CORE_PROPOSAL.md"
            )
        resolved[d] = ctx.coord(d) + idx[d]
    return resolved


# --- synchronization (async / semaphores) -----------------------------------


class Semaphore:
    """A NOC synchronization semaphore.

    On device, semaphores order the async data-movement / compute threads of a
    multi-thread kernel. The functional simulator runs a kernel body straight
    through in program order on a single thread, so waits are always already
    satisfied and set/inc/wait are no-ops kept only so async kernels referencing
    them run. `await sem` resolves immediately (SIMULATOR_SPEC §non-goals: no
    performance/ordering model).
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


def _drive_async(result):
    """Run an async kernel body to completion synchronously.

    An `async def` body returns a coroutine when called; native execution
    otherwise leaves it un-awaited (a silent no-op). Every awaitable the
    simulator exposes (`SimBlock`, `SimLazyTensor`, `Semaphore`) resolves without
    yielding, so a coroutine runs straight to `StopIteration` on the first
    `send`; the loop covers any awaitable that does suspend.
    """
    if inspect.iscoroutine(result):
        try:
            while True:
                result.send(None)
        except StopIteration:
            return
    elif inspect.isasyncgen(result):
        # `async def` bodies that use `yield` model a producer/consumer split
        # across concurrently-scheduled threads; faithfully interleaving them
        # needs an ordering model the functional simulator deliberately omits.
        result.aclose()
        raise NotImplementedError(
            "async-generator kernels (`async def` with `yield` for multi-thread "
            "producer/consumer handoff) are not supported by the simulator; use "
            "`await` without `yield`, or run on device"
        )


# --- driver -----------------------------------------------------------------


def run_kernel(sim_fn, lazy_args, scalar_args, grid):
    """Execute `sim_fn` once per grid core with the sim builtins bound.

    Outputs are `SimLazyTensor`s whose backing `SimTensor` is mutated in place
    by `remote_store`, so the caller reads results straight off them afterward.
    An `async def` kernel body is driven to completion (see `_drive_async`).
    """
    gy, gx = grid
    prev = getattr(_local, "core", None)
    try:
        for cy in range(gy):
            for cx in range(gx):
                _local.core = CoreContext(cy, cx)
                _drive_async(sim_fn(*lazy_args, *scalar_args))
    finally:
        _local.core = prev
