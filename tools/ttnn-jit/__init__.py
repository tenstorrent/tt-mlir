# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


from ttnn_jit.api import (
    jit,
)

# Hide ttnn import behind a lazy import for now if needed
_lazy = {}


def __getattr__(name):
    if name in _lazy:
        mod = import_module(_lazy[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(name)


__all__ = [
    "jit",
]
