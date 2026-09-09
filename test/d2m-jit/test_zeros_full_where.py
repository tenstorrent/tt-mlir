# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`d2m.zeros(L)`, `d2m.full(L, v)`, and `d2m.where(cond, t, f)`."""

import torch
import d2m_jit as d2m


def _make_layout():
    return d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )


def test_zeros():
    out = d2m.zeros(_make_layout()).to_host()
    assert (out == 0).all(), "zeros output had non-zero elements"


def test_full():
    # full() lowers to d2m.tile_fill, which routes the scalar through the
    # SFPU's vFloat (fp19: 1 sign + 8 exp + 10 mantissa) on Wormhole. Values
    # whose lower 13 mantissa bits are non-zero (e.g. 3.14) get truncated.
    # Other d2m fill tests dodge this by picking fp19-exact values
    # (test/python/golden/d2m/test_constants.py uses 0, 1, 1.25).
    out = d2m.full(_make_layout(), 3.14).to_host()
    expected = torch.full((64, 64), 3.14, dtype=torch.float32)
    diff = (expected - out).abs().max().item()
    assert diff < 0.01, f"full(3.14) max diff {diff}"


@d2m.kernel
def k_where_abs(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            cond = gtz(x)
            r = where(cond, x, negative(x))
            remote_store(out_t, [m_off + m, n_off + n], r)


def test_where_implements_abs():
    """`where(x > 0, x, -x)` should match `abs(x)` elementwise."""
    t = torch.randn(64, 64, dtype=torch.float32)
    L = _make_layout()
    out = d2m.empty(L)
    k_where_abs(d2m.to_layout(t, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = t.abs()
    diff = (expected - result).abs().max().item()
    assert diff < 0.01, f"where-as-abs max diff {diff}"
