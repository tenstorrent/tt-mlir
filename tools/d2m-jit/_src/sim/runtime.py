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

import inspect
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
