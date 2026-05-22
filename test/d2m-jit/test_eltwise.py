# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Block-level elementwise via _eltwise_block:
  - sigmoid as a free-function call.
  - Chained method-style: a.add(b).sigmoid()."""

import torch
import d2m_jit as d2m


@d2m.kernel
def sig_kernel(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            shard = remote_load(in_t, [m_off + m, n_off + n])
            result = sigmoid(shard)
            remote_store(out_t, [m_off + m, n_off + n], result)


@d2m.kernel
def chained_kernel(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            r = a.add(b).sigmoid()
            remote_store(out, [m_off + m, n_off + n], r)


def make_layout():
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )


def test_sigmoid_free_function():
    t = torch.randn(64, 64, dtype=torch.float32) * 0.5
    L = make_layout()
    in_d = d2m.to_layout(t, L)
    out_d = d2m.empty(L)
    sig_kernel(in_d, out_d, 1, 1, grid=(2, 2))
    result = out_d.to_host()
    expected = torch.sigmoid(t)
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"sigmoid: max abs diff {diff} too large"


def test_chained_method_call():
    """`a.add(b).sigmoid()` exercises both the method-form dispatch and
    visit_Attribute walking through a Call receiver."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    L = make_layout()
    out = d2m.empty(L)
    chained_kernel(d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = torch.sigmoid(lhs + rhs)
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"a.add(b).sigmoid(): max abs diff {diff} too large"
