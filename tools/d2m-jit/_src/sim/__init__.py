# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Torch-backed simulator backend for d2m_jit.

`install(namespace)` overwrites the host-API names in `api.py`'s module globals
with the simulator implementations, so `import d2m_jit` transparently runs
kernels in torch when `config.simulator` is set. See SIMULATOR_SPEC.md.
"""

from __future__ import annotations

import contextlib

from . import ops  # noqa: F401  (attaches SimBlock dunders/methods on import)
from .block import SimBlock
from .ops import SIM_OPS
from .runtime import Semaphore, core_index, remote_load, remote_store
from . import host

# Everything a `@d2m.kernel` body can reference, injected into the kernel's
# globals at dispatch time (SIMULATOR_SPEC §5.1). Superset check lives in the
# sim tests against D2MCompiler._syntax.
SIM_BUILTINS = dict(SIM_OPS)
SIM_BUILTINS.update(
    {
        "core_index": core_index,
        "remote_load": remote_load,
        "remote_store": remote_store,
        "Semaphore": Semaphore,
    }
)

# Class-form kernel-body ops: the compiler registers these in D2MCompiler._syntax
# as "<prefix>.<method>" (prefix starts with "!"). This maps each prefix to the
# simulator class that must provide the same methods; the sim coverage guard
# (test_simulator.py) checks it covers every "!"-prefixed group. `!tensor.*` are
# the TensorBlock method/dunder ops, mirrored on SimBlock (attached by ops.py).
SIM_CLASS_OPS = {
    "!d2m.semaphore": Semaphore,
    "!tensor": SimBlock,
}


def _make_kernel():
    def kernel(fn):
        return host.SimCompiledKernel(fn, SIM_BUILTINS)

    return kernel


# Names the simulator provides in place of the real builder. `reduction_layout`
# is pure Layout math and stays the real (already-imported) implementation.
_HOST_EXPORTS = {
    "to_layout": host.to_layout,
    "tilize": host.tilize,
    "untilize": host.untilize,
    "empty": host.empty,
    "zeros": host.zeros,
    "full": host.full,
    "arange": host.arange,
    "view": host.view,
    "view_layout": host.view_layout,
    "permute": host.permute,
    "reshape": host.reshape,
    "to_host": host.to_host,
    "LazyTensor": host.SimLazyTensor,
}


def install(namespace: dict):
    """Bind the simulator host API + kernel decorator into `namespace`."""
    namespace.update(_HOST_EXPORTS)
    namespace["kernel"] = _make_kernel()
    return namespace


def sim_kernel(fn_or_kernel):
    """Wrap a raw function (or an existing CompiledKernel) as a sim kernel.

    Accepts either a plain kernel function or a `CompiledKernel`/`SimCompiledKernel`
    (whose `.fn` is unwrapped), so a parity driver can re-home a device-built
    kernel onto the simulator without redefining it.
    """
    fn = getattr(fn_or_kernel, "fn", fn_or_kernel)
    return host.SimCompiledKernel(fn, SIM_BUILTINS)


@contextlib.contextmanager
def simulator_backend():
    """Temporarily bind the simulator host API over the live `d2m_jit` namespaces.

    Lets a single process run a kernel through the simulator even when it was
    imported in device mode: rebinds `d2m_jit`/`d2m_jit.api` host-API names to
    the sim implementations for the duration of the block, then restores them.
    This is what the sim-vs-device parity harness uses (SIMULATOR_SPEC §12).
    """
    import d2m_jit
    import d2m_jit.api as _api

    names = list(_HOST_EXPORTS) + ["kernel"]
    targets = [_api.__dict__, d2m_jit.__dict__]
    saved = [{n: ns.get(n) for n in names if n in ns} for ns in targets]
    try:
        for ns in targets:
            install(ns)
        yield
    finally:
        for ns, snap in zip(targets, saved):
            for n in names:
                if n in snap:
                    ns[n] = snap[n]
