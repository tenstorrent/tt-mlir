# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Row-wise softmax prefill kernels — Milestone M0.

Two forms:
  * `softmax_within_tile`  — each 32-wide row fits in one tile.
  * `softmax_row_wide`     — the row spans N tiles; three passes (max, sum,
                             normalize), the building block for SDPA.

Status: buildable today. reduce_max / reduce_sum / broadcast / exp / recip
are all implemented and on-device tested (incl. multi-tile-single-core).

Constraint: the reduced axis (the row) must live on one core.
  # ⛔ NEEDS[XREDUCE]: cross-core reduction, to shard a single softmax row
  across the grid. Today the row dimension must not be grid-sharded.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m


@d2m.kernel
def softmax_within_tile(x, out, m_blocks, n_blocks):
    # Numerically-stable softmax where each row is one tile wide.
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            row_max = reduce_max(t, -1)          # per-row max (R reduction)
            t_exp = exp(t - row_max)             # row_max broadcasts on use
            row_sum = reduce_sum(t_exp, -1)
            remote_store(out, [m_off + m, n_off + n], t_exp * recip(row_sum))


@d2m.kernel
def softmax_row_wide(x, neg_inf_tile, out, m_blocks, n_blocks):
    # Row spans n_blocks tiles. Standard flash-style three passes; pass 3
    # recomputes exp rather than scratch-storing it (L1 budget).
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    NEG_INF = remote_load(neg_inf_tile, [0, 0])
    # ⛔ NEEDS[INIT]: full_tile(value=-1e30) would replace this host tile.

    for m in range(m_blocks):
        # Pass 1: running row max across N tiles.
        row_max = NEG_INF
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            row_max = maximum(row_max, reduce_max(t, -1))

        # Pass 2: row sum of shifted exp.
        row_sum = reduce_sum(exp(remote_load(x, [m_off + m, n_off]) - row_max), -1)
        for n in range(1, n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            row_sum = row_sum + reduce_sum(exp(t - row_max), -1)
        row_inv = recip(row_sum)

        # Pass 3: normalise and write.
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], exp(t - row_max) * row_inv)
