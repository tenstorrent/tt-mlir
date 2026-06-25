# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""d2m-jit simulator shadow surface.

Drop-in replacement for `import d2m_jit as d2m` that runs kernels as regular
Python on top of torch -- no MLIR context, no pass pipeline, no device:

    import d2m_jit.sim as d2m

The kernel-authoring surface (`@d2m.kernel`, `core_index`, `remote_load`,
eltwise / reduction / matmul ops) and the host surface (`Layout`,
`to_layout`, `empty`, `zeros`, `full`, `tilize`, `untilize`, `view`,
`view_layout`, `permute`, `reduction_layout`, `to_host`) match the device
package. See SIMULATOR_SPEC.md.

Unlike the device package this module imports without `_ttmlir_runtime`, so it
works in environments with no tt-metal build.
"""

# Pure descriptor + dtype constants, reused unchanged from the device package.
from d2m_jit._src.tensor_layout import (  # noqa: F401
    Layout,
    float32,
    float16,
    bfloat16,
)
from d2m_jit._src.config import config  # noqa: F401

# Simulator host + kernel surface.
from d2m_jit._src.sim import (  # noqa: F401
    SimTensor,
    SimBlock,
    kernel,
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

__all__ = [
    "Layout",
    "float32",
    "float16",
    "bfloat16",
    "config",
    "SimTensor",
    "SimBlock",
    "kernel",
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
]
