# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU feed-forward block — Milestone M2 (also the MoE expert body, M7).

    y = (silu(x @ W_gate) * (x @ W_up)) @ W_down

The FLOP-dominant block of a dense prefill layer (~8·S·d^2). Three matmuls
with the silu·mul activation fused between them, stitched by views so nothing
round-trips DRAM between stages.

Status: buildable today. matmul (multi-K), silu, mul are all implemented.
  # ⛔ NEEDS[ACC-INIT]: matmul accumulates into its out-param, so each GEMM's
  output must be pre-filled with d2m.zeros (see _matmul_block TODO). An
  auto-init would let us drop the zeros prefills below.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m


@d2m.kernel
def matmul_kernel(a, b, out, m_blocks, n_blocks, k_blocks):
    # out must be pre-filled with zeros by the driver (ACC-INIT).
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            acc = remote_load(out, [m_off + m, n_off + n])
            for k in range(k_blocks):
                lhs = remote_load(a, [m_off + m, k])
                rhs = remote_load(b, [k, n_off + n])
                acc = acc + (lhs @ rhs)          # multi-K accumulation
            remote_store(out, [m_off + m, n_off + n], acc)


@d2m.kernel
def swiglu_act(gate, up, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            g = remote_load(gate, [m_off + m, n_off + n])
            u = remote_load(up, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], g.silu() * u)


def swiglu_ffn(x_lt, Wg, Wu, Wd, L_hidden, L_out, grid, dims):
    """Host driver. dims = (m, n_hidden, n_out, k_in, k_hidden) in tile units."""
    m, n_h, n_o, k_in, k_h = dims

    gate = d2m.zeros(L_hidden)            # ⛔ NEEDS[ACC-INIT]
    matmul_kernel(x_lt, Wg, gate, m, n_h, k_in, grid=grid)

    up = d2m.zeros(L_hidden)             # ⛔ NEEDS[ACC-INIT]
    matmul_kernel(x_lt, Wu, up, m, n_h, k_in, grid=grid)

    act = d2m.empty(L_hidden)
    swiglu_act(gate, up, act, m, n_h, grid=grid)   # reads gate/up as views

    y = d2m.zeros(L_out)                 # ⛔ NEEDS[ACC-INIT]
    matmul_kernel(act, Wd, y, m, n_o, k_h, grid=grid)
    return y
