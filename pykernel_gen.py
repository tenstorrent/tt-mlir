from pykernel.kernel_ast import *
import functools


matmul_template = {
    "blocking_factors": [1, 1, 1],  # | lambda | "auto" | "automatic"
    "indexing_maps": [
        lambda d0, d1, d2: (d0, d2),
        lambda d0, d1, d2: (d2, d1),
        lambda d0, d1, d2: (d0, d1),
    ],
    "iterator_types": [
        "parallel",
        "parallel",
        "reduction",
    ],
}


# TODO:
# - TensorShard = CircularBuffer, or some better name

class CB:
    def __init__(self):
        self.dtype = "Float32"
        self.tilized_shape = [4, 4]
        self.shape = [128, 128]



def pykernel_gen(blocking_factors=None, indexing_maps=None, iterator_types=None):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            threads = f(*args, **kwargs)
            if type(threads) is not list:
                threads = [threads]

            for thread in threads:
                thread(CB(), CB(), CB())

        return _wrapper
    return _decorator


@pykernel_gen(**matmul_template)
def matmul(lhs, rhs, out):
    @compute()
    def mm(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
    ):
        out = cb_in0 + cb_in1
        yield cb_out

    return mm


matmul(None, None, None)


@pykernel_gen(
    blocking_factors=[1, 1, 1],  # | lambda | "auto"
    indexing_maps=[
        lambda d0, d1, d2: (d0, d2),
        lambda d0, d1, d2: (d2, d1),
        lambda d0, d1, d2: (d0, d1),
    ],
    iterator_types=[
        "parallel",
        "parallel",
        "reduction",
    ],
)
def matmul(lhs, rhs, out):
    @compute()
    def mm(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
    ):
        cb_out += cb_in0 @ cb_in1
        yield cb_out

    return mm


@pykernel_gen(
    blocking_factors=[1, 1, 1],
    indexing_maps=[
        lambda d0, d1, d2: (d0, d2),
        lambda d0, d1, d2: (d2, d1),
        lambda d0, d1, d2: (d0, d1),
    ],
    iterator_types="explicit",
)
def matmul(lhs, rhs, out):
    @compute()
    async def mm(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
    ):
        await (cb_in0, cb_in1)
        cb_out = cb_in0 @ cb_in1
        yield cb_out

    @datamovement()
    def dm0(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
    ):
        if core_index(1) == 0:
            tx = dma(lhs.indexing_map(0), cb_in0)
            dma_wait(tx)
            semaphore_wait(rec_ready, 1, reset=0)
            tx = dma(cb_in0, cb_in0, core=(1, core_index(1)), mcast=(1, 1))
            dma_wait(tx)
            semaphore_set(sent, 1, core=(1, core_index(1)), mcast=(1, 1))
        else:
            semaphore_inc(rec_ready, 1, core=(0, core_index(1)))
            semaphore_wait(sent, 1, reset=0)
        yield cb_in0

    return (mm, dm0, dm1)


@pykernel_gen(
    blocking_factors="explicit",
    indexing_maps="explicit",
    iterator_types="explicit",
)
def matmul(lhs, rhs, out):
    M = 4
    N = 4
    K = 4

    @compute()
    async def mm(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
    ):
        for m in range(M):
            for k in range(K):
                await cb_in0
                for n in range(N):
                    await cb_in1
                    cb_out += cb_in0 @ cb_in1
                    yield cb_out

    @datamovement()
    async def dm_out(
        cb_out: CircularBuffer,
    ):
        for m in range(M):
            for k in range(K):
                for n in range(N):
                    await cb_out

    @datamovement()
    def dm0(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
    ):
        for m in range(M):
            for k in range(K):
                if core_index(1) == 0:
                    tx = dma(lhs[m, k], cb_in0)
                    dma_wait(tx)
                    semaphore_wait(rec_ready, 1, reset=0)
                    tx = dma(cb_in0, cb_in0, core=(1, core_index(1)), mcast=(1, 1))
                    dma_wait(tx)
                    semaphore_set(sent, 1, core=(1, core_index(1)), mcast=(1, 1))
                else:
                    semaphore_inc(rec_ready, 1, core=(0, core_index(1)))
                    semaphore_wait(sent, 1, reset=0)
                yield cb_in0

    return (mm, dm0, dm1)
