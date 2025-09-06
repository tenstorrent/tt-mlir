# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.kernel_ast import *
import functools
import torch
import sys


def assert_pcc(golden, actual, threshold=0.99):
    combined = torch.stack([golden.flatten(), actual.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold}\ngolden:\n{golden}\nactual:\n{actual}"


def test_add():
    @pykernel_gen(**eltwise_template)
    def add(lhs, rhs, out):
        @compute()
        def add_kernel(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
        ):
            yield lhs_shard + rhs_shard

        return Program(add_kernel)(lhs, rhs, out)

    lhs = torch.randn(128, 128)
    rhs = torch.randn(128, 128)
    out = torch.empty(128, 128)
    add(lhs, rhs, out)

    golden = lhs + rhs
    assert_pcc(golden, out)


def test_matmul():
    @pykernel_gen(**matmul_template)
    def matmul(lhs, rhs, out):
        @compute()
        def mm(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
        ):
            out = lhs_shard @ rhs_shard
            yield out

        return Program(mm)(lhs, rhs, out)

    lhs = torch.randn(128, 128)
    rhs = torch.randn(128, 128)
    out = torch.zeros(128, 128)
    matmul(lhs, rhs, out)

    golden = lhs @ rhs
    assert_pcc(golden, out)


def test_eltwise_fused():
    @pykernel_gen(**eltwise_fused_template(args=4))
    def eltwise_fused(lhs, rhs, bias, out):
        @compute()
        def kernel(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            bias_shard: Tensor,
            out_shard: Tensor,
        ):
            out = lhs_shard * rhs_shard
            out = out + bias_shard

            yield out

        return Program(kernel)(lhs, rhs, bias, out)

    lhs = torch.randn(128, 128)
    rhs = torch.randn(128, 128)
    bias = torch.randn(128, 128)
    out = torch.zeros(128, 128)
    eltwise_fused(lhs, rhs, bias, out)

    golden = lhs * rhs + bias
    assert_pcc(golden, out)


def test_matmul_fused():
    @pykernel_gen(**matmul_fused_template(args=4))
    def matmul(lhs, rhs, bias, out):
        @compute()
        def mm(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            bias_shard: Tensor,
            out_shard: Tensor,
        ):
            out = lhs_shard @ rhs_shard

            out = out + bias_shard

            yield out

        return Program(mm)(lhs, rhs, bias, out)

    lhs = torch.randn(128, 128)
    rhs = torch.randn(128, 128)
    bias = torch.randn(128, 128)
    out = torch.zeros(128, 128)
    matmul(lhs, rhs, bias, out)

    golden = lhs @ rhs
    assert_pcc(golden, out)


def custom_dm():
    @pykernel_gen(
        block_factors=[
            (1, 1),
            (1, 1),
            (1, 1),
        ],
        grid=(2, 2),
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

        M = block_factors[0][0]
        N = block_factors[1][1]
        K = block_factors[0][1] * 2

        GY = grid[0]
        GX = grid[1]

        lhs_stream = Stream(lhs)
        rhs_stream = Stream(rhs)

        @compute()
        async def mm(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
            lhs_receiver_ready: Semaphore,
            lhs_sender_sent: Semaphore,
            rhs_receiver_ready: Semaphore,
            rhs_sender_sent: Semaphore,
        ):
            for m in range(M):
                for k in range(K):
                    await lhs_shard
                    for n in range(N):
                        await rhs_shard
                        out_shard = lhs_shard @ rhs_shard
                        yield out_shard
                        # await out_shard # this is hacky for now

        @datamovement()
        async def dm0(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
            lhs_receiver_ready: Semaphore,
            lhs_sender_sent: Semaphore,
            rhs_receiver_ready: Semaphore,
            rhs_sender_sent: Semaphore,
        ):
            cy = core_index(0)
            cx = core_index(1)
            for m in range(M):
                for k in range(K):
                    if cx == 0:
                        tx = dma(lhs_stream[cy * M + m, k], lhs_shard)
                        tx.wait()
                        lhs_receiver_ready.wait(1, reset=0)
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
                    yield lhs_shard

        @datamovement()
        async def dm1(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
            lhs_receiver_ready: Semaphore,
            lhs_sender_sent: Semaphore,
            rhs_receiver_ready: Semaphore,
            rhs_sender_sent: Semaphore,
        ):
            cy = core_index(0)
            cx = core_index(1)
            for m in range(M):
                for k in range(K):
                    for n in range(N):
                        if cy == 0:
                            tx = dma(rhs_stream[k, cx * N + n], rhs_shard)
                            tx.wait()
                            rhs_receiver_ready.wait(1, reset=0)
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
                        yield rhs_shard

        return Program(mm, dm0, dm1)(lhs, rhs, out)

    lhs = torch.randn(128, 128)
    rhs = torch.randn(128, 128)
    out = torch.empty(128, 128)
    matmul(lhs, rhs, out)

    golden = lhs @ rhs
    assert_pcc(golden, out)


custom_dm()


def others():
    @pykernel_gen(
        grid=(1, 1),  # | lambda | "auto"
        block_factors=[1, 1, 1],  # | lambda | "auto"
        indexing_maps=[
            lambda m, n, k: (m, k),
            lambda m, n, k: (k, n),
            lambda m, n, k: (m, n),
        ],
        iterator_types=[
            "parallel",
            "parallel",
            "reduction",
        ],
    )
    def matmul(lhs, rhs, out):
        @compute()
        async def mm(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
        ):
            out_shard += lhs_shard @ rhs_shard
            yield out_shard

        return mm

    @pykernel_gen(
        grid=(1, 1),  # | lambda | "auto"
        block_factors=[1, 1, 1],
        indexing_maps=[
            lambda m, n, k: (m, k),
            lambda m, n, k: (k, n),
            lambda m, n, k: (m, n),
        ],
        iterator_types=None,
    )
    def matmul(lhs, rhs, out):
        @compute()
        async def mm(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
            out_shard: Tensor,
        ):
            await (lhs_shard, rhs_shard)
            out_shard = lhs_shard @ rhs_shard
            yield out_shard

        @datamovement()
        async def dm0(
            lhs_shard: Tensor,
            rhs_shard: Tensor,
        ):
            if core_index(1) == 0:
                tx = dma(lhs.indexing_map(0), lhs_shard)
                dma_wait(tx)
                semaphore_wait(rec_ready, 1, reset=0)
                tx = dma(lhs_shard, lhs_shard, core=(1, core_index(1)), mcast=(1, 1))
                dma_wait(tx)
                semaphore_set(sent, 1, core=(1, core_index(1)), mcast=(1, 1))
            else:
                semaphore_inc(rec_ready, 1, core=(0, core_index(1)))
                semaphore_wait(sent, 1, reset=0)
            yield lhs_shard

        return (mm, dm0, dm1)
