# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""d2m-jit: a Python DSL that JIT-compiles kernels through the D2M dialect.

The canonical device surface lives in `d2m_jit.api` and is re-exported from this
package, but it is resolved **lazily** (PEP 562 module `__getattr__`) rather than
star-imported at module scope. Importing a submodule runs its parent package
first, so an eager `from d2m_jit.api import *` here would drag the MLIR bindings
and the runtime extension into `import d2m_jit.sim` -- which is meant to work in
environments with no tt-metal build. Touching any device-surface attribute
(`d2m_jit.kernel`, `d2m_jit.config`, ...) imports `api` on first use.
"""

import importlib

_api_module = None


def _api():
    global _api_module
    if _api_module is None:
        _api_module = importlib.import_module("d2m_jit.api")
    return _api_module


def _api_public_names():
    return [name for name in vars(_api()) if not name.startswith("_")]


def __getattr__(name):
    # `from d2m_jit import *` looks up `__all__` through here; forward it so the
    # star-import behaves as it did when api was star-imported eagerly.
    if name == "__all__":
        return _api_public_names()
    # Never import api to answer an interpreter dunder probe (`__path__`,
    # `__getstate__`, ...) -- that would defeat the laziness.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module 'd2m_jit' has no attribute {name!r}")
    try:
        return getattr(_api(), name)
    except AttributeError:
        raise AttributeError(f"module 'd2m_jit' has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(_api_public_names()))
