# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Float tile reductions exposed through the d2m-jit block DSL."""

import torch
import d2m_jit as d2m


@d2m.kernel
def k_reduce_sum_cols(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], reduce_sum(x, 1))


@d2m.kernel
def k_reduce_max_rows(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], x.reduce_max(0))


def _make_layout():
    return d2m.Layout(
        shape=(32, 32),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )


def _run(kernel, tensor):
    layout = _make_layout()
    out = d2m.empty(layout)
    kernel(d2m.to_layout(tensor, layout), out, 1, 1, grid=(1, 1))
    return out.to_host()


def test_reduce_sum_cols():
    tensor = torch.arange(32, dtype=torch.float32).reshape(32, 1).repeat(1, 32)
    tensor = tensor / 100.0
    result = _run(k_reduce_sum_cols, tensor)

    expected = tensor.sum(dim=1)
    diff = (expected - result[:, 0]).abs().max().item()
    assert diff < 0.05, f"reduce_sum(dim=1): max diff {diff}"


def test_reduce_max_rows_method_form():
    row_bias = torch.linspace(-0.5, 0.5, 32, dtype=torch.float32).reshape(32, 1)
    col_values = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(1, 32)
    tensor = row_bias + col_values
    result = _run(k_reduce_max_rows, tensor)

    expected = tensor.max(dim=0).values
    diff = (expected - result[0, :]).abs().max().item()
    assert diff < 0.05, f"reduce_max(dim=0): max diff {diff}"
