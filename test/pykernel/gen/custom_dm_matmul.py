# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.d2m_api import *
from utils import assert_pcc
import torch


@d2m_jit()
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


@d2m_jit()
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
    lhs = torch.randn(512, 512)
    rhs = torch.randn(512, 512)
    out = torch.zeros(512, 512)
    grid = (2, 2)
    block_shape = [1, 1]
    m_blocks = (lhs.shape[0] // 32) // block_shape[0] // grid[0]
    n_blocks = (lhs.shape[1] // 32) // block_shape[1] // grid[1]
    add(
        TensorLayout(lhs, block_shape, grid_shape=[2, 2]),
        TensorLayout(rhs, block_shape, grid_shape=[2, 2]),
        TensorLayout(out, block_shape, grid_shape=[2, 2]),
        m_blocks,
        n_blocks,
        grid=grid,
    )

    golden = lhs + rhs
    assert_pcc(golden, out)


test_eltwise()


def test_matmul():
    lhs = torch.randn(128, 128)
    rhs = torch.randn(128, 128)
    out = torch.zeros(128, 128)
    GY = 2
    GX = 2
    matmul(
        TensorLayout(lhs, [2, 2]),
        TensorLayout(rhs, [2, 2]),
        TensorLayout(out, [2, 2]),
        2,
        2,
        2,
        GY,
        GX,
        grid=(GY, GX),
    )

    golden = lhs @ rhs
    assert_pcc(golden, out)
