# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python / torch simulator backend for d2m-jit (shadow mode).

This subpackage runs a `@d2m.kernel` body as regular Python, backing the
block-level tile ops with `torch`. It has no dependency on an MLIR `Context`,
the pass pipeline, or the `_ttmlir_runtime` device extension -- it only reuses
the pure `Layout` descriptor.

See SIMULATOR_SPEC.md for the design. Import via the shadow module:

    import d2m_jit.sim as d2m
"""

from .tensors import SimTensor, SimBlock
from .host import (
    to_layout,
    empty,
    zeros,
    full,
    tilize,
    untilize,
    view,
    view_layout,
    permute,
    reduction_layout,
    to_host,
)
from .run import kernel

__all__ = [
    "SimTensor",
    "SimBlock",
    "to_layout",
    "empty",
    "zeros",
    "full",
    "tilize",
    "untilize",
    "view",
    "view_layout",
    "permute",
    "reduction_layout",
    "to_host",
    "kernel",
]
