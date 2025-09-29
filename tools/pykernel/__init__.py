# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

from pykernel._src.kernel_types import CircularBuffer, Kernel, CompileTimeValue

from pykernel.api import (
    ttkernel_compile,
    compute_thread,
    reader_thread,
    writer_thread,
    ttkernel_tensix_compile,
    ttkernel_noc_compile,
)

import pykernel.d2m_api as d2m_api

# Hide ttnn import behind a lazy import for now.
# `import pykernel` will not import ttnn, but `from pykernel import PykernelOp` will
_lazy = {"PyKernelOp": "pykernel._src.kernel_op"}


def __getattr__(name):
    if name in _lazy:
        mod = import_module(_lazy[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(name)


__all__ = [
    "ttkernel_compile",
    "compute_thread",
    "reader_thread",
    "writer_thread",
    "ttkernel_tensix_compile",
    "ttkernel_noc_compile",
    "CircularBuffer",
    "Kernel",
    "CompileTimeValue",
    "PyKernelOp",
]
