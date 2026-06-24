# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""One-tile SDPA bring-up kernels for D2M-JIT work.

This computes:

    softmax(Q @ K^T) @ V

for a single 32x32 tile shard. The key input is expected to already be laid
out as K^T. The three-kernel chain is kept as a comparison baseline;
``sdpa_1tile_fused`` is the bring-up target and emits QK^T, row-softmax, and
PV inside one D2M-JIT kernel.
"""

import d2m_jit as d2m


@d2m.kernel
def sdpa_qk_1tile(q, k_t, scores):
    q_tile = remote_load(q, [0, 0])
    kt_tile = remote_load(k_t, [0, 0])
    scores_tile = q_tile @ kt_tile

    remote_store(scores, [0, 0], scores_tile)


@d2m.kernel
def sdpa_softmax_1tile_fused(scores, probs):
    scores_tile = remote_load(scores, [0, 0])
    row_max = reduce_max(scores_tile, 1)
    numer = exp(scores_tile - row_max)
    denom = reduce_sum(numer, 1)
    probs_tile = numer / denom

    remote_store(probs, [0, 0], probs_tile)


@d2m.kernel
def sdpa_pv_1tile(probs, v, out):
    probs_tile = remote_load(probs, [0, 0])
    v_tile = remote_load(v, [0, 0])
    result = probs_tile @ v_tile

    remote_store(out, [0, 0], result)


@d2m.kernel
def sdpa_1tile_fused(q, k_t, v, out):
    q_tile = remote_load(q, [0, 0])
    kt_tile = remote_load(k_t, [0, 0])
    v_tile = remote_load(v, [0, 0])

    scores = q_tile @ kt_tile
    row_max = reduce_max(scores, 1)
    numer = exp(scores - row_max)
    denom = reduce_sum(numer, 1)
    probs = numer / denom
    result = probs @ v_tile

    remote_store(out, [0, 0], result)


@d2m.kernel
def sdpa_grid_1tile_fused(q, k_t, v, out):
    m = core_index(0)
    n = core_index(1)
    q_tile = remote_load(q, [m, n])
    kt_tile = remote_load(k_t, [m, n])
    v_tile = remote_load(v, [m, n])

    scores = q_tile @ kt_tile
    row_max = reduce_max(scores, 1)
    numer = exp(scores - row_max)
    denom = reduce_sum(numer, 1)
    probs = numer / denom
    result = probs @ v_tile

    remote_store(out, [m, n], result)


@d2m.kernel
def sdpa_full_2x2_fused(q, k_t, v, out):
    row = core_index(0)
    value_col = core_index(1)

    q0 = remote_load(q, [row, 0])
    q1 = remote_load(q, [row, 1])

    kt00 = remote_load(k_t, [0, 0])
    kt10 = remote_load(k_t, [1, 0])
    kt01 = remote_load(k_t, [0, 1])
    kt11 = remote_load(k_t, [1, 1])

    scores0 = (q0 @ kt00) + (q1 @ kt10)
    scores1 = (q0 @ kt01) + (q1 @ kt11)

    row_max = reduce_max_pair(scores0, scores1, 1)

    numer0 = exp(scores0 - row_max)
    numer1 = exp(scores1 - row_max)
    denom = reduce_sum_pair(numer0, numer1, 1)

    probs0 = numer0 / denom
    probs1 = numer1 / denom

    v0 = remote_load(v, [0, value_col])
    v1 = remote_load(v, [1, value_col])
    result = (probs0 @ v0) + (probs1 @ v1)

    remote_store(out, [row, value_col], result)


@d2m.kernel
def sdpa_block_2x2_fused(q, k_t, v, out):
    head_y = core_index(0)
    head_x = core_index(1)

    q_block = remote_load(q, [head_y, head_x])
    kt_block = remote_load(k_t, [head_y, head_x])
    v_block = remote_load(v, [head_y, head_x])

    scores = q_block @ kt_block
    row_max = reduce_max(scores, 1)
    numer = exp(scores - row_max)
    denom = reduce_sum(numer, 1)
    probs = numer / denom
    result = probs @ v_block

    remote_store(out, [head_y, head_x], result)


@d2m.kernel
def sdpa_block_fused(q, k_t, v, out):
    head_y = core_index(0)
    head_x = core_index(1)

    q_block = remote_load(q, [head_y, head_x])
    kt_block = remote_load(k_t, [head_y, head_x])
    v_block = remote_load(v, [head_y, head_x])

    scores = q_block @ kt_block
    row_max = reduce_max(scores, 1)
    numer = exp(scores - row_max)
    denom = reduce_sum(numer, 1)
    probs = numer / denom
    result = probs @ v_block

    remote_store(out, [head_y, head_x], result)
