# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Binary tile comparisons: `eq`, `ne`, `gt`, `ge`, `lt`, `le`.

Each comparison writes 1.0/0.0 into its output tile (same element type as
the inputs). Combined with `where`, this gives masked computation.

Coverage:
 - Free-function form (`d2m.eq/gt/ge(a, b)`).
 - Method form (`a.ne/lt/le(b)`).
 - Every comparison in the family (`eq`, `ne`, `gt`, `ge`, `lt`, `le`) is
   exercised in at least one form.
 - One end-to-end mask use (`where(a.ge(b), a, b)` ≡ `maximum(a, b)`).
"""

import torch
import d2m_jit as d2m


def _make_layout():
    return d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )


@d2m.kernel
def k_ge_free(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], ge(a, b))


@d2m.kernel
def k_lt_method(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a.lt(b))


@d2m.kernel
def k_eq_free(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], eq(a, b))


@d2m.kernel
def k_ne_method(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a.ne(b))


@d2m.kernel
def k_gt_free(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], gt(a, b))


@d2m.kernel
def k_le_method(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a.le(b))


@d2m.kernel
def k_max_via_where_ge(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            cond = a.ge(b)
            remote_store(out, [m_off + m, n_off + n], where(cond, a, b))


def _run_binary(kernel, lhs, rhs):
    L = _make_layout()
    out = d2m.empty(L)
    kernel(d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out, 1, 1, grid=(2, 2))
    return out.to_host()


def _assert_compare_matches(expected, out, name):
    """Tolerate a small fraction of disagreeing cells near the equality
    boundary: on Wormhole the SFPU rounds f32 through fp19 (1 sign + 8 exp +
    10 mantissa), so inputs that differ only in the lower 13 mantissa bits
    compare as equal on device but not in torch. Rather than reconstruct
    `lhs`/`rhs` here, we simply allow a global mismatch rate below 0.5%,
    which is well above the expected fp19 boundary-rounding rate but far
    below what a genuine op bug would produce."""
    diff_cells = expected != out
    n_diff = diff_cells.sum().item()
    assert n_diff < out.numel() * 0.005, (
        f"{name}: {n_diff}/{out.numel()} mismatches (>0.5%); "
        f"likely a real bug, not fp19 rounding"
    )


def test_ge_free_function():
    """`d2m.ge(a, b)` matches `(a >= b).float()` elementwise."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    out = _run_binary(k_ge_free, lhs, rhs)
    expected = (lhs >= rhs).to(torch.float32)
    _assert_compare_matches(expected, out, "ge")


def test_lt_method_form():
    """`a.lt(b)` matches `(a < b).float()` elementwise."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    out = _run_binary(k_lt_method, lhs, rhs)
    expected = (lhs < rhs).to(torch.float32)
    _assert_compare_matches(expected, out, "lt")


def _make_eq_inputs():
    """lhs/rhs that share ~half their cells exactly (so eq/ne see both
    True and False), with the rest distinct."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = lhs.clone()
    # Perturb roughly half the cells so they no longer compare equal.
    mask = torch.rand(64, 64) < 0.5
    rhs[mask] = torch.randn(64, 64, dtype=torch.float32)[mask]
    return lhs, rhs


def test_eq_free_function():
    """`d2m.eq(a, b)` matches `(a == b).float()` elementwise."""
    lhs, rhs = _make_eq_inputs()
    out = _run_binary(k_eq_free, lhs, rhs)
    expected = (lhs == rhs).to(torch.float32)
    _assert_compare_matches(expected, out, "eq")


def test_ne_method_form():
    """`a.ne(b)` matches `(a != b).float()` elementwise."""
    lhs, rhs = _make_eq_inputs()
    out = _run_binary(k_ne_method, lhs, rhs)
    expected = (lhs != rhs).to(torch.float32)
    _assert_compare_matches(expected, out, "ne")


def test_gt_free_function():
    """`d2m.gt(a, b)` matches `(a > b).float()` elementwise."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    out = _run_binary(k_gt_free, lhs, rhs)
    expected = (lhs > rhs).to(torch.float32)
    _assert_compare_matches(expected, out, "gt")


def test_le_method_form():
    """`a.le(b)` matches `(a <= b).float()` elementwise."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    out = _run_binary(k_le_method, lhs, rhs)
    expected = (lhs <= rhs).to(torch.float32)
    _assert_compare_matches(expected, out, "le")


def test_ge_as_maximum_via_where():
    """`where(a >= b, a, b)` ≡ `maximum(a, b)` -- exercises the typical
    use-case where a compare feeds a `where`."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    out = _run_binary(k_max_via_where_ge, lhs, rhs)
    expected = torch.maximum(lhs, rhs)
    # fp19 SFPU rounding on Wormhole: ~2^-9 relative error per element.
    diff = (expected - out).abs().max().item()
    assert diff < 0.01, f"where(a>=b, a, b) vs maximum: max diff {diff}"
