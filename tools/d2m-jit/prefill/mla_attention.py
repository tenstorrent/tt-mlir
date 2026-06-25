# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 MLA + sparse attention prefill — Milestone M8.

Multi-head Latent Attention with a sparse "lightning indexer" and a KV
compressor, as exercised by tt-xla's
`tests/torch/models/deepseek_v4/test_deepseek_v4_tp.py`.

Structure:
    c_kv      = x @ W_DKV                     # KV down-projection (latent)
    c_q       = x @ W_DQ                      # Q down-projection (latent)
    K_C, V    = c_kv @ W_UK, c_kv @ W_UV      # up-projections
    Q_C       = c_q @ W_UQ
    q_R, k_R  = RoPE(c_q @ W_QR), RoPE(x @ W_KR)   # decoupled RoPE
    sel       = top_k(indexer(c_q, x), n_sel) # lightning indexer (sparse)
    scores    = (Q_C·K_C + q_R·k_R) over selected tokens
    O         = softmax(scores) @ V ; out = O @ W_O

Green today: all the latent/up/down projections (multi-K matmul), RoPE,
softmax, the score/PV matmuls. Blocked / DeepSeek-specific:
  # ⛔ NEEDS[TOPK]: the lightning indexer selects the top-n_sel KV tokens per
  query. No on-device top-k; this is the sparse-attention gate.
  # ⛔ NEEDS[GATHER]: gathering the selected KV rows for the score matmul.
  # ⛔ NEEDS[ABSORB]: the inference-time absorption fusion (W_UK -> W_UQ,
  W_UV -> W_O) folds two matmuls into one against a host-premultiplied weight,
  so K/V are never materialized. Best expressed as a compile-time pattern that
  reads the constant weights via from_value and emits one smaller matmul.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m

from .swiglu_ffn import matmul_kernel
from .rope import apply_rope
from .sdpa_prefill import sdpa_prefill


def mla_projections(x_lt, W, L, grid, dims):
    """Latent down- + up-projections. All are the standard multi-K GEMM."""
    m, _ = dims["mn"]
    # c_kv = x @ W_DKV ; c_q = x @ W_DQ
    c_kv = d2m.zeros(L["c_kv"]); matmul_kernel(x_lt, W["DKV"], c_kv, *dims["dkv"], grid=grid)
    c_q = d2m.zeros(L["c_q"]); matmul_kernel(x_lt, W["DQ"], c_q, *dims["dq"], grid=grid)

    # ⛔ NEEDS[ABSORB]: instead of materializing K_C = c_kv @ W_UK and
    # Q_C = c_q @ W_UQ separately, fold (W_UQ @ W_UK^T) at compile time and
    # attend in latent space. Shown unfused here:
    K_C = d2m.zeros(L["kc"]); matmul_kernel(c_kv, W["UK"], K_C, *dims["uk"], grid=grid)
    V = d2m.zeros(L["v"]);   matmul_kernel(c_kv, W["UV"], V, *dims["uv"], grid=grid)
    Q_C = d2m.zeros(L["qc"]); matmul_kernel(c_q, W["UQ"], Q_C, *dims["uq"], grid=grid)
    return c_kv, c_q, K_C, V, Q_C


def lightning_indexer(c_q, x_lt, W, L, grid, dims):
    """Lightweight per-(query, kv) scores -> top-n_sel KV selection."""
    # idx_scores = indexer(c_q, x)  (a small matmul + weights_proj) — green.
    # sel = top_k(idx_scores, n_sel)
    # ⛔ NEEDS[TOPK]: no on-device top-k. For the testbed, compute `sel`
    # host-side from idx_scores.to_host() and feed the selection in.
    raise NotImplementedError("NEEDS[TOPK]: lightning-indexer top-k selection")


def mla_attention(x_lt, W, masks, L, grid, dims, compress=False):
    """Full MLA prefill block. `compress` toggles the KV compressor layer."""
    c_kv, c_q, K_C, V, Q_C = mla_projections(x_lt, W, L, grid, dims)

    # Decoupled RoPE on the rope-carrying query/key projections (green).
    q_R = apply_rope(c_q, W["cos"], W["sin_signed"], L["q_r"], grid, *dims["mn"])
    # k_R = apply_rope(x @ W_KR, ...)  — same shape.

    if compress:
        # KV compressor: a learned low-rank reduction of the cache.
        # If implemented as a matmul/pooling it's green; if it pools across
        # tokens (a reduction over the sequence axis) it needs that axis on
        # one core (see XREDUCE).  # ⛔ NEEDS[XREDUCE] (conditional)
        pass

    # Sparse selection of KV tokens via the lightning indexer.
    sel = lightning_indexer(c_q, x_lt, W, L, grid, dims)  # ⛔ NEEDS[TOPK]
    # ⛔ NEEDS[GATHER]: gather K_C / k_R / V rows at `sel` for the score matmul.

    # Score + softmax + PV over the (gathered) selected tokens. With ABSORB
    # this attends in latent space; shown here as a standard SDPA over the
    # gathered K_C/V plus the decoupled-RoPE score term.
    out = sdpa_prefill(Q_C, K_C, V, masks["causal"], masks["neg_inf"],
                       L["scores"], L["attn_out"], grid, dims["attn"])
    # out = out @ W_O   (ACC-INIT zeros) — elided.
    return out, q_R, sel
