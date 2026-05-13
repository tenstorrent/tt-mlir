# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: d2m-jit

import d2m_jit as d2m
from utils import assert_pcc, arange_tile
import torch


@d2m.kernel
def matmul(lhs, rhs, out, K, M, N, GY, GX):
    cy = core_index(0)
    cx = core_index(1)
    for k in range(K):
        for m in range(M):
            lhs_shard = remote_load(
                lhs, [m, k], mcast_start_index=[cy, 0], mcast_shape=[1, GX]
            )
            for n in range(N):
                rhs_shard = remote_load(
                    rhs, [k, n], mcast_start_index=[0, cx], mcast_shape=[GY, 1]
                )
                out_shard = lhs_shard + rhs_shard
                remote_store(out, [m, n], out_shard)


@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_offset = core_index(0) * m_blocks
    n_offset = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            lhs_shard = remote_load(lhs, [m_offset + m, n_offset + n])
            rhs_shard = remote_load(rhs, [m_offset + m, n_offset + n])
            out_shard = lhs_shard + rhs_shard
            remote_store(out, [m_offset + m, n_offset + n], out_shard)


def test_eltwise():
    lhs = arange_tile(512, 512, dtype=torch.float)
    rhs = arange_tile(512, 512, dtype=torch.float)
    grid = (2, 2)
    block_shape = [1, 1]
    m_blocks = (lhs.shape[0] // 32) // block_shape[0] // grid[0]
    n_blocks = (lhs.shape[1] // 32) // block_shape[1] // grid[1]

    L_in = d2m.Layout(
        shape=lhs.shape, dtype=lhs.dtype, block_shape=block_shape, grid_shape=[8, 8]
    )
    L_out = d2m.Layout(
        shape=lhs.shape, dtype=lhs.dtype, block_shape=block_shape, grid_shape=[2, 2]
    )

    lhs_d = d2m.to_layout(lhs, L_in)
    rhs_d = d2m.to_layout(rhs, L_in)
    out_d = d2m.empty(L_out)
    add(lhs_d, rhs_d, out_d, m_blocks, n_blocks, grid=grid)
    out = out_d.to_host()

    print(out[::32, ::32])
    golden = lhs + rhs
    assert_pcc(golden, out)


def test_eltwise2():
    lhs = arange_tile(64, 64, dtype=torch.float)
    rhs = arange_tile(64, 64, dtype=torch.float)
    grid = (1, 1)
    block_shape = [1, 1]
    m_blocks = (lhs.shape[0] // 32) // block_shape[0] // grid[0]
    n_blocks = (lhs.shape[1] // 32) // block_shape[1] // grid[1]

    L = d2m.Layout(
        shape=lhs.shape, dtype=lhs.dtype, block_shape=block_shape, grid_shape=[1, 1]
    )

    lhs_d = d2m.to_layout(lhs, L)
    rhs_d = d2m.to_layout(rhs, L)
    out_d = d2m.empty(L)
    add(lhs_d, rhs_d, out_d, m_blocks, n_blocks, grid=grid)
    out = out_d.to_host()

    print(out[::32, ::32])
    golden = lhs + rhs
    assert_pcc(golden, out)


test_eltwise()
