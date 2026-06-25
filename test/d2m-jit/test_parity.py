# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-vs-device PCC parity.

Each test builds the same kernel graph and runs it on both backends via the
`config.backend` switch, asserting the outputs agree by PCC (`assert_parity`).
The simulator is the intended-semantics oracle, so a parity failure flags a
device lowering regression.

These need a device (they run the device backend), so the whole module is
marked `parity` and skipped when the d2m-jit runtime extension is unavailable.
Run just these with `pytest -m parity`; skip them with `pytest -m 'not parity'`.

Only kernels where device and sim are expected to agree live here. Cases where
the device deliberately diverges (raw `empty` contents, matmul into `empty`,
multicast) are not parity-testable -- see SIMULATOR_SPEC.md §9.
"""

import pytest
import torch

import d2m_jit as d2m
from utils import assert_parity, device_runtime_available

pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(
        not device_runtime_available(),
        reason="d2m-jit device runtime not available",
    ),
]


def _layout(shape, block_shape=(1, 1), grid=(1, 1), dtype=d2m.float32):
    return d2m.Layout(
        shape=shape, dtype=dtype, block_shape=list(block_shape), grid_shape=list(grid)
    )


# --- kernels -----------------------------------------------------------------


@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)


@d2m.kernel
def fused_exp_add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(lhs, [m_off + m, n_off + n])
            y = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], x.exp() + y)


@d2m.kernel
def softmax_row(x, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            row_max = reduce_max(t, 1)
            e = exp(t - row_max)
            row_sum = reduce_sum(e, 1)
            remote_store(out, [m_off + m, n_off + n], e * recip(row_sum))


@d2m.kernel
def reduce_sum_cols(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, 0], reduce_sum(x, 1))


@d2m.kernel
def matmul_kernel(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a @ b)


# --- parity tests ------------------------------------------------------------


def test_parity_eltwise_add():
    def build():
        layout = _layout((64, 64), grid=(2, 2))
        out = d2m.empty(layout)  # fully overwritten by stores
        add(
            d2m.to_layout(torch.randn(64, 64), layout),
            d2m.to_layout(torch.randn(64, 64), layout),
            out,
            1,
            1,
            grid=(2, 2),
        )
        return out.to_host()

    assert_parity(build)


def test_parity_fused_exp_add():
    def build():
        layout = _layout((64, 64), grid=(2, 2))
        out = d2m.empty(layout)
        fused_exp_add(
            d2m.to_layout(torch.randn(64, 64), layout),
            d2m.to_layout(torch.randn(64, 64), layout),
            out,
            1,
            1,
            grid=(2, 2),
        )
        return out.to_host()

    assert_parity(build)


def test_parity_softmax_within_tile():
    def build():
        layout = _layout((32, 32))
        out = d2m.empty(layout)
        softmax_row(d2m.to_layout(torch.randn(32, 32), layout), out, 1, 1, grid=(1, 1))
        return out.to_host()

    assert_parity(build)


def test_parity_reduce_sum_cols():
    def build():
        layout = _layout((64, 32), grid=(2, 1))
        out = d2m.empty(d2m.reduction_layout(layout, 1))
        reduce_sum_cols(
            d2m.to_layout(torch.randn(64, 32), layout), out, 1, 1, grid=(2, 1)
        )
        return out.to_host()

    assert_parity(build)


def test_parity_matmul_via_zeros():
    def build():
        layout = _layout((64, 64), grid=(2, 2))
        out = d2m.zeros(layout)  # accumulator prefill -- correct on device
        matmul_kernel(
            d2m.to_layout(torch.randn(64, 64), layout),
            d2m.to_layout(torch.randn(64, 64), layout),
            out,
            1,
            1,
            grid=(2, 2),
        )
        return out.to_host()

    assert_parity(build)
