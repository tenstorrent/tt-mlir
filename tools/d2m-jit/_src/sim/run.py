# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SimKernel: run a `@d2m.kernel` body as regular Python, once per simulated
core of the grid.

The body is the user's original function. We rebuild it against a globals dict
that injects the in-kernel op namespace (`core_index`, `remote_load`, eltwise,
reductions, matmul, ...), preserving the original closure so int captures still
resolve. Control flow, index arithmetic, and helper calls then just execute --
that is the whole "runs as regular Python" idea.
"""

import functools
import inspect
import types

from . import ops
from .tensors import SimTensor


def _drive_async(result):
    """Run an async kernel body to completion synchronously.

    An `async def` body returns a coroutine when called; native execution
    otherwise leaves it un-awaited (a silent no-op). Every awaitable the sim
    exposes (`SimBlock`, `SimTensor`, `Semaphore`) resolves without yielding, so
    a coroutine runs straight to `StopIteration` on the first `send`; the loop
    covers any awaitable that does suspend. A plain (non-async) body returns its
    value directly and is left untouched.
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
        # needs an ordering model the functional sim deliberately omits.
        result.aclose()
        raise NotImplementedError(
            "async-generator kernels (`async def` with `yield` for multi-thread "
            "producer/consumer handoff) are not supported by the simulator; use "
            "`await` without `yield`, or run on device"
        )


class SimKernel:
    """Wraps a user kernel function for host (sim) execution."""

    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self.fn = fn
        # Globals for the running body: original module globals first, op
        # namespace last so op names win over anything else in scope.
        self._sim_globals = {**fn.__globals__, **ops.SIM_OPS}
        self._runnable = types.FunctionType(
            fn.__code__,
            self._sim_globals,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )

    def __call__(
        self,
        *args,
        grid,
        num_outs: int = 1,
        block_factors=None,
        indexing_maps=None,
        iterator_types=None,
        kernel_io_in_dram=None,
    ):
        if indexing_maps is not None or iterator_types is not None:
            raise NotImplementedError(
                "sim does not support declarative generic forms "
                "(indexing_maps / iterator_types) yet; use the core_index "
                "SPMD form"
            )

        self._validate_args(args, num_outs)

        if len(grid) != 2:
            raise NotImplementedError("sim supports 2-D grids only")
        gy, gx = int(grid[0]), int(grid[1])

        try:
            for y in range(gy):
                for x in range(gx):
                    ops._set_current_core((y, x))
                    _drive_async(self._runnable(*args))
        finally:
            ops._set_current_core(None)

        # Outputs are mutated in place (via remote_store), matching the device
        # CompiledKernel which also returns None.
        return None

    def _validate_args(self, args, num_outs):
        lazy = 0
        saw_scalar = False
        for i, a in enumerate(args):
            if isinstance(a, SimTensor):
                if saw_scalar:
                    raise TypeError(
                        f"argument {i} to kernel '{self.fn.__name__}' is a "
                        f"SimTensor but a scalar was already seen; tensor "
                        f"arguments must precede scalars"
                    )
                lazy += 1
            elif isinstance(a, int) and not isinstance(a, bool):
                saw_scalar = True
            else:
                raise TypeError(
                    f"argument {i} to kernel '{self.fn.__name__}' has "
                    f"unsupported type {type(a).__name__}: {a!r}; kernel "
                    f"arguments must be SimTensor or int"
                )
        if num_outs < 1:
            raise ValueError(f"num_outs must be >= 1 (got {num_outs})")
        if lazy < num_outs:
            raise ValueError(
                f"kernel call has {lazy} tensor args; need at least "
                f"{num_outs} for outputs"
            )


def kernel(fn):
    """Decorate a user function as a simulated d2m_jit kernel."""
    return SimKernel(fn)
