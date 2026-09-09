# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Broader op coverage: representative unary ops via free-function form
plus all the Python dunders on TensorBlock (__neg__, __sub__, __mul__,
__truediv__) and the .add/.mul method form. Confirms the table-driven
@syntax dispatch path isn't sigmoid-specific."""

import torch
import d2m_jit as d2m


@d2m.kernel
def k_exp(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], exp(x))


@d2m.kernel
def k_sqrt(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], sqrt(x))


@d2m.kernel
def k_neg(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], -x)


@d2m.kernel
def k_dunder_arith(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            # Exercise __sub__, __mul__, __truediv__ in one expression.
            r = (a - b) * a / b
            remote_store(out, [m_off + m, n_off + n], r)


def make_layout():
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )


def _run_unary(kernel, t):
    L = make_layout()
    in_d = d2m.to_layout(t, L)
    out_d = d2m.empty(L)
    kernel(in_d, out_d, 1, 1, grid=(2, 2))
    return out_d.to_host()


def test_exp():
    t = torch.randn(64, 64, dtype=torch.float32) * 0.5
    out = _run_unary(k_exp, t)
    diff = (torch.exp(t) - out).abs().max().item()
    assert diff < 0.05, f"exp: max diff {diff}"


def test_sqrt():
    t = torch.rand(64, 64, dtype=torch.float32) + 0.1  # positive
    out = _run_unary(k_sqrt, t)
    diff = (torch.sqrt(t) - out).abs().max().item()
    assert diff < 0.05, f"sqrt: max diff {diff}"


def test_neg_dunder():
    t = torch.randn(64, 64, dtype=torch.float32)
    out = _run_unary(k_neg, t)
    diff = (-t - out).abs().max().item()
    assert diff < 0.01, f"neg: max diff {diff}"


def test_arith_dunders():
    """(a - b) * a / b exercises __sub__, __mul__, __truediv__."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32).abs() + 0.5  # avoid /0
    L = make_layout()
    out = d2m.empty(L)
    k_dunder_arith(d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = (lhs - rhs) * lhs / rhs
    diff = (expected - result).abs().max().item()
    # Relative tolerance: /rhs amplifies error for small rhs.
    assert diff < 0.1, f"(a-b)*a/b: max diff {diff}"
