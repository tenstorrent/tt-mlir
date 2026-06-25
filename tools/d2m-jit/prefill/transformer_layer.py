# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single dense transformer prefill layer — Milestone M3.

Composes the M0–M2 kernels into one decoder layer:

    h  = x + Attn(RMSNorm(x))
    y  = h + FFN(RMSNorm(h))

This is the first *end-to-end prefill layer*. At toy width it is buildable
today with the standing workarounds (host causal mask, zeros-prefill matmuls).
M4 stacks N of these via a host loop with DRAM staging between layers.

The embedding lookup that precedes layer 0 is the one piece with no on-device
path today:
  # ⛔ NEEDS[GATHER]: data-dependent row gather (embedding / indexed_row_copy).
  Worked around by doing the embedding host-side and feeding activations in.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m

from .rmsnorm import rmsnorm_layer
from .swiglu_ffn import swiglu_ffn
from .sdpa_prefill import sdpa_prefill


@d2m.kernel
def residual_add(a, b, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            xa = remote_load(a, [m_off + m, n_off + n])
            xb = remote_load(b, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], xa + xb)


def attention(x_lt, weights, masks, L, grid, dims):
    """Wq/Wk/Wv/Wo projections (matmul_kernel) + RoPE + sdpa_prefill.

    Elided projection matmuls call swiglu_ffn.matmul_kernel; RoPE calls
    rope.apply_rope on Q/K. Shown as a single sdpa_prefill call for brevity —
    the projections are the same multi-K GEMM as the FFN, pre-zeroed.
    """
    # q = matmul_kernel(x, Wq); k = ...; v = ...   (ACC-INIT zeros each)
    # q = apply_rope(q, ...); k = apply_rope(k, ...)
    attn_out = sdpa_prefill(
        weights["q"], weights["k"], weights["v"],
        masks["causal"], masks["neg_inf"],
        L["scores"], L["attn_out"], grid, dims["attn"],
    )
    # o = matmul_kernel(attn_out, Wo)  (ACC-INIT zeros)
    return attn_out


def transformer_layer(x_lt, weights, masks, L, grid, dims, eps):
    """One decoder layer. x_lt: tokens x d_model (host-embedded for layer 0)."""
    m, n = dims["mn"]

    # --- Attention sub-block ---
    normed = rmsnorm_layer(x_lt, weights["attn_norm"], eps, L["io"], grid, m, n)
    attn = attention(normed, weights, masks, L, grid, dims)
    h = d2m.empty(L["io"])
    residual_add(x_lt, attn, h, m, n, grid=grid)

    # --- FFN sub-block ---
    normed2 = rmsnorm_layer(h, weights["ffn_norm"], eps, L["io"], grid, m, n)
    ffn = swiglu_ffn(normed2, weights["Wg"], weights["Wu"], weights["Wd"],
                     L["hidden"], L["io"], grid, dims["ffn"])
    y = d2m.empty(L["io"])
    residual_add(h, ffn, y, m, n, grid=grid)
    return y


def prefill(tokens_embedded_lt, layers, masks, L, grid, dims, eps):
    """Milestone M4: stack N layers. Host loop; views chain within a layer,
    DRAM staging between layers (activations exceed L1 at real width)."""
    x = tokens_embedded_lt           # ⛔ NEEDS[GATHER]: embedding done host-side
    for layer_w in layers:
        x = transformer_layer(x, layer_w, masks, L, grid, dims, eps)
    # final norm + LM head (matmul) elided; same primitives as above.
    return x
