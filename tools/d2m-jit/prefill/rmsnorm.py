# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm prefill kernel — Milestone M0.

    y = x * gamma * rsqrt(mean(x^2, axis=-1) + eps)

Status: buildable with today's primitives (reduce_mean / square / rsqrt /
broadcast are all implemented and on-device tested). Single-core: the
row-reduction over the feature axis must live on one core (see XREDUCE).

Spec only — see ../PREFILL_MILESTONES.md. Real primitives used: reduce_mean,
square, rsqrt, mul, add, remote_load/store. Gaps: eps as a constant tile
(INIT) — worked around by passing a host-filled eps tile as an input.
"""

import d2m_jit as d2m


@d2m.kernel
def rmsnorm(x, gamma, eps_tile, out, m_blocks, n_blocks):
    # x:        tokens x d_model, tiled, sharded
    # gamma:    1 x d_model scale (broadcast over rows)
    # eps_tile: 1x1-grid tile pre-filled with `eps` host-side.
    #           # ⛔ NEEDS[INIT]: kernel-body full_tile(value=eps) would let us
    #           drop this synthetic input and build eps in-kernel.
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    eps = remote_load(eps_tile, [0, 0])

    for m in range(m_blocks):
        # Pass 1: mean of squares across the row (axis -1 == R reduction).
        # reduce_mean already divides by the element count, so no INV_N tile.
        sum_sq = reduce_mean(square(remote_load(x, [m_off + m, n_off])), -1)
        for n in range(1, n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            sum_sq = sum_sq + reduce_mean(square(t), -1)
        # reduce result broadcasts back across the row on the next eltwise.
        inv_rms = rsqrt(sum_sq + eps)  # 1-wide row stat, bcast on use

        # Pass 2: scale and write.
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            g = remote_load(gamma, [0, n_off + n])
            remote_store(out, [m_off + m, n_off + n], t * inv_rms * g)


def rmsnorm_layer(x_lt, gamma_lt, eps, L_io, grid, m_blocks, n_blocks):
    """Host driver: allocate the eps tile and output, dispatch the kernel."""
    eps_tile = d2m.full(
        d2m.Layout(shape=(32, 32), dtype=x_lt.layout.dtype,
                   block_shape=[1, 1], grid_shape=[1, 1], tiled=True),
        eps,
    )
    out = d2m.empty(L_io)
    rmsnorm(x_lt, gamma_lt, eps_tile, out, m_blocks, n_blocks, grid=grid)
    return out
