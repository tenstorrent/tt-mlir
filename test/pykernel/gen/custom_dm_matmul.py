# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.d2m_api import *
from utils import assert_pcc
import torch


@pykernel_gen(
    block_factors=[
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    grid=(2, 2),
    # kernel_source_mode="store",
)
def matmul(lhs, rhs, out, block_factors=None, grid=None):
    assert block_factors is not None
    assert grid is not None

    # assert M
    assert block_factors[0][0] == block_factors[2][0]
    # assert K
    assert block_factors[0][1] == block_factors[1][0]
    # assert N
    assert block_factors[1][1] == block_factors[2][1]

    GY = grid[0]
    GX = grid[1]
    GK = 2

    M = block_factors[0][0]
    N = block_factors[1][1]
    K = block_factors[0][1] * GK

    lhs_stream = Stream(lhs)
    rhs_stream = Stream(rhs)

    @compute()
    async def mm(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
        lhs_receiver_ready: Semaphore,
        lhs_sender_sent: Semaphore,
        rhs_receiver_ready: Semaphore,
        rhs_sender_sent: Semaphore,
    ):
        for k in range(K):
            for m in range(M):
                lhs_shard = lhs_cb.wait()
                for n in range(N):
                    rhs_shard = rhs_cb.wait()
                    out_shard = out_cb.reserve()
                    out = lhs_shard @ rhs_shard
                    out_shard.store(out)
                    out_cb.wait()  # compute needs to clear the output

    @datamovement()
    async def dm0(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
        lhs_receiver_ready: Semaphore,
        lhs_sender_sent: Semaphore,
        rhs_receiver_ready: Semaphore,
        rhs_sender_sent: Semaphore,
    ):
        cy = core_index(0)
        cx = core_index(1)
        for k in range(K):
            for m in range(M):
                lhs_shard = lhs_cb.reserve()
                if cx == 0:
                    tx = dma(lhs_stream[cy * M + m, k], lhs_shard)
                    tx.wait()
                    lhs_receiver_ready.wait(GK - 1, reset=0)
                    tx = dma(
                        lhs_shard,
                        lhs_shard,
                        core=(cy, 1),
                        mcast=(1, GX - 1),
                    )
                    tx.wait()
                    lhs_sender_sent.set(1, core=(cy, 1), mcast=(1, GX - 1))
                else:
                    lhs_receiver_ready.inc(1, core=(cy, 0))
                    lhs_sender_sent.wait(1, reset=0)

    @datamovement()
    async def dm1(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
        lhs_receiver_ready: Semaphore,
        lhs_sender_sent: Semaphore,
        rhs_receiver_ready: Semaphore,
        rhs_sender_sent: Semaphore,
    ):
        cy = core_index(0)
        cx = core_index(1)
        for k in range(K):
            for m in range(M):
                for n in range(N):
                    rhs_shard = rhs_cb.reserve()
                    if cy == 0:
                        tx = dma(rhs_stream[k, cx * N + n], rhs_shard)
                        tx.wait()
                        rhs_receiver_ready.wait(GK - 1, reset=0)
                        tx = dma(
                            rhs_shard,
                            rhs_shard,
                            core=(1, cx),
                            mcast=(GY - 1, 1),
                        )
                        tx.wait()
                        rhs_sender_sent.set(1, core=(1, cx), mcast=(GY - 1, 1))
                    else:
                        rhs_receiver_ready.inc(1, core=(0, cx))
                        rhs_sender_sent.wait(1, reset=0)

    return Program(mm, dm0, dm1)(lhs, rhs, out)


lhs = torch.randn(128, 128)
rhs = torch.randn(128, 128)
out = torch.zeros(128, 128)
matmul(lhs, rhs, out)

golden = lhs @ rhs
assert_pcc(golden, out)
print("Passed")
