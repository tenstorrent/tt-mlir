from pykernel.kernel_ast import *
import functools
import torch


matmul_template = {
    "grid": (1, 1),  # | lambda | "auto" | "automatic"
    "block_factors": [1, 1, 1],  # | lambda | "auto" | "automatic"
    "indexing_maps": [
        lambda m, n, k: (m, k),
        lambda m, n, k: (k, n),
        lambda m, n, k: (m, n),
    ],
    "iterator_types": [
        "parallel",
        "parallel",
        "reduction",
    ],
}

explicit_template = {
    "grid": (1, 1),  # | lambda | "auto" | "automatic"
    "block_factors": None,
    "indexing_maps": None,
    "iterator_types": None,
}


@pykernel_gen(**matmul_template)
def matmul(lhs, rhs, out):
    @compute()
    def mm(
        lhs_shard: Tensor,
        rhs_shard: Tensor,
        out_shard: Tensor,
    ):
        out = lhs_shard + rhs_shard
        yield out

    return mm


lhs = torch.randn(128, 128)
rhs = torch.randn(128, 128)
out = torch.zeros(128, 128)
matmul(lhs, rhs, out)


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


@pykernel_gen(grid=(1, 1))
def matmul(lhs, rhs, out):
    M = 4
    N = 4
    K = 4

    @compute()
    async def mm(
        lhs_shard: Tensor,
        rhs_shard: Tensor,
        out_shard: Tensor,
    ):
        for m in range(M):
            for k in range(K):
                await lhs_shard
                for n in range(N):
                    await rhs_shard
                    out_shard += lhs_shard @ rhs_shard
                    yield out_shard

    @datamovement()
    async def dm_out(
        out_shard: Tensor,
    ):
        for m in range(M):
            for k in range(K):
                for n in range(N):
                    await out_shard

    @datamovement()
    def dm0(
        lhs_shard: Tensor,
        rhs_shard: Tensor,
    ):
        for m in range(M):
            for k in range(K):
                if core_index(1) == 0:
                    tx = dma(lhs[m, k], lhs_shard)
                    dma_wait(tx)
                    semaphore_wait(rec_ready, 1, reset=0)
                    tx = dma(
                        lhs_shard, lhs_shard, core=(1, core_index(1)), mcast=(1, 1)
                    )
                    dma_wait(tx)
                    semaphore_set(sent, 1, core=(1, core_index(1)), mcast=(1, 1))
                else:
                    semaphore_inc(rec_ready, 1, core=(0, core_index(1)))
                    semaphore_wait(sent, 1, reset=0)
                yield lhs_shard

    return (mm, dm0, dm1)
