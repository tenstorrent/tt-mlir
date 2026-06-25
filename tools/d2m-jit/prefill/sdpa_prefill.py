# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Scaled-dot-product attention, prefill, materialized scores — Milestone M1.

    scores = (Q @ K^T) * scale + causal_mask
    probs  = softmax(scores)            # row-wide, see softmax.py
    out    = probs @ V

This is the non-flash form: it materializes the [S, S] score block. Fine for
modest sequence lengths / single head / single core. The fused online-softmax
form for long context is flash_sdpa_prefill.py (M6).

Status: the math is buildable today — matmul(transpose_b=True) gives Q@K^T
without a physical transpose, scale is eltwise, softmax is implemented, P@V is
a matmul. The two gaps:
  # ⛔ NEEDS[CMASK]: a causal/triangular mask builder. Worked around here by
  passing a host-built mask tensor (0 / -inf) and adding it before softmax.
  # ⛔ NEEDS[ACC-INIT]: zeros-prefill the matmul accumulators.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m
from .softmax import softmax_row_wide


@d2m.kernel
def qk_scaled_masked(Q, K, mask, scores, m_blocks, n_blocks, k_blocks, scale_q):
    # scores = (Q @ K^T) * scale + mask   ; scores pre-zeroed (ACC-INIT).
    # scale_q is an index-typed arg; the real scalar scale is folded into Q
    # host-side (Q already pre-scaled) OR supplied as a 1x1 tile. Here we keep
    # it simple and assume Q is pre-scaled, then just add the mask.
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            acc = remote_load(scores, [m_off + m, n_off + n])
            for k in range(k_blocks):
                q = remote_load(Q, [m_off + m, k])
                kk = remote_load(K, [n_off + n, k])   # K stored [S, D]
                acc = acc + q.matmul(kk, transpose_b=True)  # Q @ K^T
            # ⛔ NEEDS[CMASK]: `mask` is a host-built causal tile (0 / -inf).
            msk = remote_load(mask, [m_off + m, n_off + n])
            remote_store(scores, [m_off + m, n_off + n], acc + msk)


@d2m.kernel
def pv_matmul(P, V, out, m_blocks, n_blocks, k_blocks):
    # out = P @ V ; out pre-zeroed (ACC-INIT).
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            acc = remote_load(out, [m_off + m, n_off + n])
            for k in range(k_blocks):
                p = remote_load(P, [m_off + m, k])
                v = remote_load(V, [k, n_off + n])
                acc = acc + (p @ v)
            remote_store(out, [m_off + m, n_off + n], acc)


def sdpa_prefill(Q_lt, K_lt, V_lt, mask_lt, neg_inf_lt, L_scores, L_out,
                 grid, dims):
    """Host driver. Q assumed pre-scaled by 1/sqrt(d). dims in tile units:
    (s_q, s_kv, d_blocks)."""
    s_q, s_kv, d = dims

    scores = d2m.zeros(L_scores)         # ⛔ NEEDS[ACC-INIT]
    qk_scaled_masked(Q_lt, K_lt, mask_lt, scores, s_q, s_kv, d, 0, grid=grid)

    probs = d2m.empty(L_scores)
    softmax_row_wide(scores, neg_inf_lt, probs, s_q, s_kv, grid=grid)

    out = d2m.zeros(L_out)               # ⛔ NEEDS[ACC-INIT]
    pv_matmul(probs, V_lt, out, s_q, d, s_kv, grid=grid)
    return out
