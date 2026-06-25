# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flash-attention prefill (fused online softmax) — Milestone M6.

The long-context form: never materialize the [S, S] score matrix. Walk the KV
blocks with a running max `m`, running sum `l`, and running output `O`,
applying the online-softmax correction per block.

    for kv in blocks:
        s      = (Q @ K_kv^T) * scale + causal_mask
        m_new  = maximum(m, reduce_max(s, -1))
        alpha  = exp(m - m_new)
        p      = exp(s - m_new)
        l      = alpha * l + reduce_sum(p, -1)
        O      = alpha * O + (p @ V_kv)
        m      = m_new
    O = O * recip(l)

Status: the per-step math is buildable today (matmul transpose_b, reduce_max /
reduce_sum, exp, broadcast). What's missing:
  # ⛔ NEEDS[CMASK]: per-block causal mask (host mask works, but a builder
  avoids shipping S^2 mask tiles).
  # ⛔ NEEDS[MOUT]: m / l are running state. We carry them in scf.for loop
  vars here; exposing them as extra kernel outputs (num_outs > 1) is untested
  and wanted for debugging / cross-block carry across kernel launches.
  # ⛔ NEEDS[ACC-INIT]: O and the per-block P@V accumulator need zero init.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m


@d2m.kernel
def flash_attn_block(Q, K, V, mask, O, neg_inf, m_off_const,
                     q_blocks, kv_blocks, d_blocks):
    # Per (q-block) online softmax over all kv-blocks. Q assumed pre-scaled.
    m_off = core_index(0) * q_blocks
    NEG_INF = remote_load(neg_inf, [0, 0])          # ⛔ NEEDS[INIT]

    for qb in range(q_blocks):
        # Running state for this q-block (single d-tile shown for clarity).
        m_state = NEG_INF
        l_state = NEG_INF - NEG_INF                 # 0.0  ⛔ NEEDS[INIT]
        O_acc = remote_load(O, [m_off + qb, 0])     # pre-zeroed ⛔ NEEDS[ACC-INIT]

        for kv in range(kv_blocks):
            # s = Q_qb @ K_kv^T  (+ causal mask)
            s = NEG_INF - NEG_INF
            for k in range(d_blocks):
                q = remote_load(Q, [m_off + qb, k])
                kk = remote_load(K, [kv, k])
                s = s + q.matmul(kk, transpose_b=True)
            s = s + remote_load(mask, [m_off + qb, kv])   # ⛔ NEEDS[CMASK]

            m_new = maximum(m_state, reduce_max(s, -1))
            alpha = exp(m_state - m_new)
            p = exp(s - m_new)
            l_state = alpha * l_state + reduce_sum(p, -1)

            # O = alpha * O + p @ V_kv
            pv = O_acc - O_acc                       # 0  ⛔ NEEDS[ACC-INIT]
            for k in range(d_blocks):
                v = remote_load(V, [kv, k])
                pv = pv + (p @ v)
            O_acc = alpha * O_acc + pv
            m_state = m_new

        remote_store(O, [m_off + qb, 0], O_acc * recip(l_state))
