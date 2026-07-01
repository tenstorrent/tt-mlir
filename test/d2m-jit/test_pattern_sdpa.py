# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import d2m_jit as d2m

from d2m_jit.patterns.sdpa_1tile_to_kernel import (
    sdpa_block_2x2_fused,
    sdpa_1tile_fused,
    sdpa_full_2x2_fused,
    sdpa_grid_1tile_fused,
    sdpa_pv_1tile,
    sdpa_qk_1tile,
    sdpa_softmax_1tile_fused,
)
from utils import assert_pcc


@d2m.kernel
def sdpa_full_qk_2x2(q, k_t, scores):
    row = core_index(0)
    col = core_index(1)

    q0 = remote_load(q, [row, 0])
    kt0 = remote_load(k_t, [0, col])
    q1 = remote_load(q, [row, 1])
    kt1 = remote_load(k_t, [1, col])

    remote_store(scores, [row, col], (q0 @ kt0) + (q1 @ kt1))


@d2m.kernel
def sdpa_full_row_max_first(scores, row_max):
    row = core_index(0)
    scores_tile = remote_load(scores, [row, 0])
    remote_store(row_max, [row, 0], reduce_max(scores_tile, 1))


@d2m.kernel
def sdpa_full_row_max_accumulate(scores, acc_max, row_max):
    row = core_index(0)
    scores_tile = remote_load(scores, [row, 1])
    acc_tile = remote_load(acc_max, [row, 0])
    remote_store(row_max, [row, 0], acc_tile.maximum(reduce_max(scores_tile, 1)))


@d2m.kernel
def sdpa_full_exp_scores(scores, row_max, numer):
    row = core_index(0)
    col = core_index(1)

    scores_tile = remote_load(scores, [row, col])
    max_tile = tile_bcast_col(remote_load(row_max, [row, 0]))
    remote_store(numer, [row, col], exp(scores_tile - max_tile))


@d2m.kernel
def sdpa_full_row_sum_first(numer, row_sum):
    row = core_index(0)
    numer_tile = remote_load(numer, [row, 0])
    remote_store(row_sum, [row, 0], reduce_sum(numer_tile, 1))


@d2m.kernel
def sdpa_full_row_sum_accumulate(numer, acc_sum, row_sum):
    row = core_index(0)
    numer_tile = remote_load(numer, [row, 1])
    acc_tile = remote_load(acc_sum, [row, 0])
    remote_store(row_sum, [row, 0], acc_tile + reduce_sum(numer_tile, 1))


@d2m.kernel
def sdpa_full_normalize(numer, row_sum, probs):
    row = core_index(0)
    col = core_index(1)

    numer_tile = remote_load(numer, [row, col])
    sum_tile = tile_bcast_col(remote_load(row_sum, [row, 0]))
    remote_store(probs, [row, col], numer_tile / sum_tile)


@d2m.kernel
def sdpa_full_pv_2x2(probs, v, out):
    row = core_index(0)
    col = core_index(1)

    p0 = remote_load(probs, [row, 0])
    v0 = remote_load(v, [0, col])
    p1 = remote_load(probs, [row, 1])
    v1 = remote_load(v, [1, col])

    remote_store(out, [row, col], (p0 @ v0) + (p1 @ v1))


def _make_layout():
    return d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )


def test_sdpa_1tile_kernel_chain_on_device():
    torch.manual_seed(0)
    q = torch.randn(32, 32, dtype=torch.float32) * 0.25
    k = torch.randn(32, 32, dtype=torch.float32) * 0.25
    v = torch.randn(32, 32, dtype=torch.float32)

    layout = _make_layout()
    scores_d = d2m.empty(layout)
    probs_d = d2m.empty(layout)
    out_d = d2m.empty(layout)

    sdpa_qk_1tile(
        d2m.to_layout(q, layout),
        d2m.to_layout(k.transpose(0, 1).contiguous(), layout),
        scores_d,
        grid=(1, 1),
    )
    sdpa_softmax_1tile_fused(scores_d, probs_d, grid=(1, 1))
    sdpa_pv_1tile(probs_d, d2m.to_layout(v, layout), out_d, grid=(1, 1))

    expected = torch.softmax(q @ k.transpose(0, 1), dim=-1) @ v
    assert_pcc(expected, out_d.to_host(), threshold=0.99)


def test_sdpa_1tile_fused_kernel_on_device():
    torch.manual_seed(0)
    q = torch.randn(32, 32, dtype=torch.float32) * 0.25
    k = torch.randn(32, 32, dtype=torch.float32) * 0.25
    v = torch.randn(32, 32, dtype=torch.float32)

    layout = _make_layout()
    out_d = d2m.empty(layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        sdpa_1tile_fused(
            d2m.to_layout(q, layout),
            d2m.to_layout(k.transpose(0, 1).contiguous(), layout),
            d2m.to_layout(v, layout),
            out_d,
            grid=(1, 1),
        )

        expected = torch.softmax(q @ k.transpose(0, 1), dim=-1) @ v
        assert_pcc(expected, out_d.to_host(), threshold=0.99)
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul


def test_sdpa_grid_1tile_fused_multicore_on_device():
    torch.manual_seed(0)
    grid = (2, 2)
    shape = (grid[0] * 32, grid[1] * 32)
    q = torch.randn(*shape, dtype=torch.float32) * 0.25
    k_t = torch.randn(*shape, dtype=torch.float32) * 0.25
    v = torch.randn(*shape, dtype=torch.float32)

    layout = d2m.Layout(
        shape=shape,
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=list(grid),
    )
    out_d = d2m.empty(layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        sdpa_grid_1tile_fused(
            d2m.to_layout(q, layout),
            d2m.to_layout(k_t, layout),
            d2m.to_layout(v, layout),
            out_d,
            grid=grid,
        )

        expected = torch.empty_like(q)
        for gy in range(grid[0]):
            row = slice(gy * 32, (gy + 1) * 32)
            for gx in range(grid[1]):
                col = slice(gx * 32, (gx + 1) * 32)
                q_tile = q[row, col]
                kt_tile = k_t[row, col]
                v_tile = v[row, col]
                expected[row, col] = torch.softmax(q_tile @ kt_tile, dim=-1) @ v_tile

        assert_pcc(expected, out_d.to_host(), threshold=0.99)
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul


def test_sdpa_full_2x2_multicore_on_device():
    torch.manual_seed(0)
    grid = (2, 2)
    shape = (64, 64)

    q = torch.randn(*shape, dtype=torch.float32) * 0.25
    k = torch.randn(*shape, dtype=torch.float32) * 0.25
    k_t = k.transpose(0, 1).contiguous()
    v = torch.randn(*shape, dtype=torch.float32)

    layout = d2m.Layout(
        shape=shape,
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=list(grid),
    )
    row_layout = d2m.reduction_layout(layout, 1, allow_cross_tile=True)

    scores_d = d2m.empty(layout)
    row_max0_d = d2m.empty(row_layout)
    row_max_d = d2m.empty(row_layout)
    numer_d = d2m.empty(layout)
    row_sum0_d = d2m.empty(row_layout)
    row_sum_d = d2m.empty(row_layout)
    probs_d = d2m.empty(layout)
    out_d = d2m.empty(layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        q_d = d2m.to_layout(q, layout)
        kt_d = d2m.to_layout(k_t, layout)
        v_d = d2m.to_layout(v, layout)

        sdpa_full_qk_2x2(q_d, kt_d, scores_d, grid=grid)
        sdpa_full_row_max_first(scores_d, row_max0_d, grid=(2, 1))
        sdpa_full_row_max_accumulate(scores_d, row_max0_d, row_max_d, grid=(2, 1))
        sdpa_full_exp_scores(scores_d, row_max_d, numer_d, grid=grid)
        sdpa_full_row_sum_first(numer_d, row_sum0_d, grid=(2, 1))
        sdpa_full_row_sum_accumulate(numer_d, row_sum0_d, row_sum_d, grid=(2, 1))
        sdpa_full_normalize(numer_d, row_sum_d, probs_d, grid=grid)
        sdpa_full_pv_2x2(probs_d, v_d, out_d, grid=grid)

        expected = torch.softmax(q @ k_t, dim=-1) @ v
        actual = out_d.to_host()
        assert_pcc(expected, actual, threshold=0.99)
        max_diff = (expected - actual).abs().max().item()
        assert max_diff < 0.2, f"full 2x2 SDPA max diff {max_diff} too large"
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul


def test_sdpa_full_2x2_fused_multicore_on_device():
    torch.manual_seed(0)
    grid = (2, 2)
    shape = (64, 64)

    q = torch.randn(*shape, dtype=torch.float32) * 0.25
    k = torch.randn(*shape, dtype=torch.float32) * 0.25
    k_t = k.transpose(0, 1).contiguous()
    v = torch.randn(*shape, dtype=torch.float32)

    layout = d2m.Layout(
        shape=shape,
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=list(grid),
    )
    out_d = d2m.empty(layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        sdpa_full_2x2_fused(
            d2m.to_layout(q, layout),
            d2m.to_layout(k_t, layout),
            d2m.to_layout(v, layout),
            out_d,
            grid=grid,
        )

        expected = torch.softmax(q @ k_t, dim=-1) @ v
        actual = out_d.to_host()
        assert_pcc(expected, actual, threshold=0.99)
        max_diff = (expected - actual).abs().max().item()
        assert max_diff < 0.2, f"fused full 2x2 SDPA max diff {max_diff} too large"
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul


def test_sdpa_block_2x2_fused_packed_heads_on_device():
    torch.manual_seed(0)
    grid = (2, 2)
    head_shape = (64, 64)
    shape = (grid[0] * head_shape[0], grid[1] * head_shape[1])

    q = torch.randn(*shape, dtype=torch.float32) * 0.25
    k_t = torch.randn(*shape, dtype=torch.float32) * 0.25
    v = torch.randn(*shape, dtype=torch.float32)

    layout = d2m.Layout(
        shape=shape,
        dtype=d2m.float32,
        block_shape=[2, 2],
        grid_shape=list(grid),
    )
    out_d = d2m.empty(layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        sdpa_block_2x2_fused(
            d2m.to_layout(q, layout),
            d2m.to_layout(k_t, layout),
            d2m.to_layout(v, layout),
            out_d,
            grid=grid,
        )

        expected = torch.empty_like(q)
        for gy in range(grid[0]):
            row = slice(gy * head_shape[0], (gy + 1) * head_shape[0])
            for gx in range(grid[1]):
                col = slice(gx * head_shape[1], (gx + 1) * head_shape[1])
                q_block = q[row, col]
                kt_block = k_t[row, col]
                v_block = v[row, col]
                expected[row, col] = torch.softmax(q_block @ kt_block, dim=-1) @ v_block

        actual = out_d.to_host()
        assert_pcc(expected, actual, threshold=0.99)
        max_diff = (expected - actual).abs().max().item()
        assert max_diff < 0.2, f"packed block 2x2 SDPA max diff {max_diff} too large"
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul
