# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Torch-backed simulator backend for d2m_jit.

`install(namespace)` overwrites the host-API names in `api.py`'s module globals
with the simulator implementations, so `import d2m_jit` transparently runs
kernels in torch when `config.simulator` is set. See SIMULATOR_SPEC.md.
"""

from __future__ import annotations

from . import ops  # noqa: F401  (attaches SimBlock dunders/methods on import)
from .block import SimBlock
from .ops import SIM_OPS
from .runtime import core_index, remote_load, remote_store
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
    }
)


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
