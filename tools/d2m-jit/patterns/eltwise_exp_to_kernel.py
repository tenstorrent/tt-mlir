# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lower TTIR `ttir.exp` to a fused d2m-jit kernel template.

Single-op example: matches `ttir.exp` directly and replaces it with a
d2m-jit subgraph (to_layout → @kernel call doing tile_exp → from_device).

Load this file via `--d2m-python-rewrites=path/to/this.py` (forthcoming
C++ pass) or import-and-call `d2m.apply_patterns(module)` from a host
Python driver. The pattern's emission targets the rewriter's insertion
point — no rebuild needed when this file changes.
"""

import d2m_jit as d2m
from ttmlir.dialects import ttir


# Block-level tile-exp kernel. One thread per (Y, X) grid core; each core
# sweeps `m_blocks` × `n_blocks` tile blocks via remote_load/remote_store.
# The body uses the registered unary `exp` op which wraps `d2m.tile_exp`
# in a `linalg.generic` over the tensor of tiles.
#
# Note: docstrings inside @d2m.kernel function bodies aren't supported by
# the AST visitor (string constants aren't a kernel expression).
@d2m.kernel
def exp_fused(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            shard = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], shard.exp())


@d2m.pattern(root=ttir.ExpOp, benefit=10)
def lower_exp(op, rewriter):
    """Replace one `ttir.exp` with the fused d2m subgraph.

    Layout choice: tiled L1, single-block per cell, 1x1 grid. Cheap to
    set up; correctness-first for the POC. A real lowering would pick
    grid_shape from the operand shape to spread work across cores.
    """
    src = op.operands[0]

    # Tiled L1 with `block_shape=[1,1]` (one tile per block) and
    # `grid_shape=[1,1]` (one core). For 32x32 inputs that's literally
    # one tile end-to-end.
    L = d2m.infer_layout(
        op.result,
        tiled=True,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )

    x = d2m.to_layout(d2m.from_value(src), L)
    out = d2m.empty(L)
    exp_fused(x, out, 1, 1, grid=(1, 1))
    return d2m.from_device(out)
