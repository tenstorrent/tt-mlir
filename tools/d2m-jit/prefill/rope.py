# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rotary position embedding (RoPE) prefill kernel — Milestone M0 / M1.

    rope(x) = x * cos + rotate_half(x) * sin

`rotate_half` is expressed as a *view-roll* of x plus a precomputed sign tile,
so no `concat` primitive is needed:

    rotate_half(x) = roll_view(x) * sign        # sign = [-1...,+1...]
    rope(x)        = x * cos + roll_view(x) * (sign * sin)

cos / (sign*sin) are precomputed host-side and passed as input tiles. The roll
is a free `view_layout` (metadata, no data movement).

Status: buildable today — pure eltwise + views. This is also the cleanest
auto-fusion target (see KERNEL_AUDIT discussion): a PDL pattern matching the
frontend's mul/neg/concat/add RoPE DAG can replace it with this single kernel.

Spec only — see ../PREFILL_MILESTONES.md.
"""

import d2m_jit as d2m


@d2m.kernel
def rope(x, x_rolled, cos, sin_signed, out, m_blocks, n_blocks):
    # x_rolled: a view of x produced host-side via d2m.view_layout (the
    #           half-rotation index permutation). Passed in so the kernel
    #           body stays a pure eltwise pass.
    # cos:        precomputed cos table tile(s).
    # sin_signed: precomputed (sign * sin) so rotate_half's negate folds in.
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            xx = remote_load(x, [m_off + m, n_off + n])
            xr = remote_load(x_rolled, [m_off + m, n_off + n])
            c = remote_load(cos, [m_off + m, n_off + n])
            s = remote_load(sin_signed, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], xx * c + xr * s)


def apply_rope(x_lt, cos_lt, sin_signed_lt, L_io, grid, m_blocks, n_blocks):
    """Host driver: build the half-rotation view, dispatch the kernel."""
    # ⛔ NEEDS[INIT]: cos / sin_signed could be device-built via arange + cos/sin
    # once kernel-body init helpers exist; today they are host-precomputed.
    x_rolled = d2m.view_layout(x_lt, lambda d0, d1, d2, d3: (d0, d1, d2, d3))
    # ^ placeholder lambda; the real roll permutes the feature half. View is
    #   metadata-only (is_view=True), consumed directly by the kernel.
    out = d2m.empty(L_io)
    rope(x_lt, x_rolled, cos_lt, sin_signed_lt, out, m_blocks, n_blocks, grid=grid)
    return out
