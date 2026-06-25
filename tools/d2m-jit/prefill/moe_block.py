# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MoE prefill block — Milestone M7.

    logits  = x @ W_router                      # dense matmul (green)
    probs   = softmax(logits)                   # green
    idx, w  = top_k(probs, k)                   # ⛔ NEEDS[TOPK]
    y       = sum_e  w_e * Expert_e(x)          # expert = SwiGLU (M2)

Two formulations:
  * **dense-masked** (testbed): run every expert for every token, combine
    with gate weights that are 0 for non-selected (token, expert) pairs.
    Avoids sparse dispatch entirely. Buildable today *except* the top-k that
    produces the mask — stub that host-side.
  * **sparse dispatch** (real): gather tokens per expert, run, scatter back.
    # ⛔ NEEDS[GATHER]: data-dependent gather/scatter (indexed_row_copy).

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m

from .swiglu_ffn import swiglu_ffn, matmul_kernel
from .softmax import softmax_row_wide


@d2m.kernel
def weighted_accumulate(expert_out, weight, acc, m_blocks, n_blocks):
    # acc += weight * expert_out. weight is the per-token gate weight,
    # pre-broadcast to full tiles host-side and zeroed for non-selected
    # (token, expert) pairs (this is where the top-k mask is baked in).
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            e = remote_load(expert_out, [m_off + m, n_off + n])
            w = remote_load(weight, [m_off + m, n_off + n])
            prev = remote_load(acc, [m_off + m, n_off + n])
            remote_store(acc, [m_off + m, n_off + n], prev + e * w)


def moe_router(x_lt, W_router, neg_inf_lt, L_logits, grid, dims):
    """logits = x @ W_router ; probs = softmax(logits). Both green today."""
    m, n_e, k = dims
    logits = d2m.zeros(L_logits)         # ⛔ NEEDS[ACC-INIT]
    matmul_kernel(x_lt, W_router, logits, m, n_e, k, grid=grid)
    probs = d2m.empty(L_logits)
    softmax_row_wide(logits, neg_inf_lt, probs, m, n_e, grid=grid)
    # ⛔ NEEDS[TOPK]: top_k(probs, k) -> (expert_ids, gate_weights). No on-device
    # primitive; for the testbed compute it from probs.to_host() and feed the
    # masked, pre-broadcast gate-weight tensors into weighted_accumulate below.
    return probs


def moe_block_dense_masked(x_lt, experts, gate_weights, L, grid, dims):
    """Dense-masked MoE: every expert over every token, gate-weighted sum.

    gate_weights[e]: per-token weight for expert e, 0 where not in top-k,
    pre-broadcast host-side (top-k mask baked in -> dodges TOPK on device).
    """
    m, n = dims["mn"]
    acc = d2m.zeros(L["io"])             # ⛔ NEEDS[ACC-INIT]
    for e, expert_w in enumerate(experts):
        y_e = swiglu_ffn(x_lt, expert_w["Wg"], expert_w["Wu"], expert_w["Wd"],
                         L["hidden"], L["io"], grid, dims["ffn"])
        weighted_accumulate(y_e, gate_weights[e], acc, m, n, grid=grid)
    return acc
