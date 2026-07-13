# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Loop-carried eltwise accumulator (`acc = zeros([...]); for k: acc += ...`).

A runtime `for` loop becomes an `scf.for` whose `acc` is a loop-carried iter_arg;
`acc += <eltwise>` accumulates in place via `__add_acc__` (the eltwise dual of
`__matmul_acc__`). The accumulator stays L1-resident and the compute loop
DST-chunks it, so it SCALES past DST capacity (8 f32 tiles): a 4x4 (16-tile) and
6x6 (36-tile) accumulator both accumulate correctly.

Crucially the accumulator must be COMPUTE-initialized (`zeros(...)`), NOT
DM-seeded: `acc = remote_load(...)` then `acc += ...` makes the accumulator a
cross-thread circular-buffer FIFO that silently miscomputes, so the tracer
rejects it with a clear error (see test_dm_seeded_accumulator_rejected).
"""

import pytest
import torch
import d2m_jit as d2m
from d2m_jit._src.errors import D2mJitError
from utils import assert_pcc

N_ITERS = 5


@d2m.kernel
def _acc_1x1(in0, out):
    acc = zeros([1, 1])
    for k in range(5):
        acc += remote_load(in0, [0, 0])
    remote_store(out, [0, 0], acc)


@d2m.kernel
def _acc_4x4(in0, out):
    acc = zeros([4, 4])
    for k in range(5):
        acc += remote_load(in0, [0, 0])
    remote_store(out, [0, 0], acc)


@d2m.kernel
def _acc_6x6(in0, out):
    acc = zeros([6, 6])
    for k in range(5):
        acc += remote_load(in0, [0, 0])
    remote_store(out, [0, 0], acc)


# 1x1 (fits DST), 4x4 = 16 tiles (2x DST), 6x6 = 36 tiles (4.5x DST). The last
# two confirm the accumulator stays L1-resident and DST-chunks (scales past DST).
@pytest.mark.parametrize("kernel,R", [(_acc_1x1, 1), (_acc_4x4, 4), (_acc_6x6, 6)])
def test_eltwise_loop_accumulator_scales(kernel, R):
    d2m.mesh((1, 1), topology=("linear", "linear"))
    Li = d2m.Layout(shape=(R * 32, R * 32), dtype=d2m.float32,
                    block_shape=[R, R], grid_shape=[1, 1])
    Lo = d2m.Layout(shape=(R * 32, R * 32), dtype=d2m.float32,
                    block_shape=[R, R], grid_shape=[1, 1])
    fi = torch.randn(R * 32, R * 32, dtype=torch.float32)
    si = d2m.to_layout(fi, Li)
    so = d2m.empty(Lo)
    kernel(si, so, grid=(1, 1))
    result = so.to_host()
    assert_pcc(N_ITERS * fi, result)


def test_dm_seeded_accumulator_rejected():
    """A DM-seeded loop-carried accumulator must be rejected at trace time with a
    clear error (it would otherwise silently miscompute on device)."""

    @d2m.kernel
    def _bad(in0, zin, out):
        acc = remote_load(zin, [0, 0])   # DM seed -- not allowed for an accumulator
        for k in range(4):
            acc += remote_load(in0, [0, 0])
        remote_store(out, [0, 0], acc)

    d2m.mesh((1, 1), topology=("linear", "linear"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32,
                   block_shape=[1, 1], grid_shape=[1, 1])
    fi = torch.randn(32, 32, dtype=torch.float32)
    zr = torch.zeros(32, 32, dtype=torch.float32)
    si = d2m.to_layout(fi, L)
    zi = d2m.to_layout(zr, L)
    so = d2m.empty(L)
    with pytest.raises(D2mJitError, match="loop-carried accumulator"):
        _bad(si, zi, so, grid=(1, 1))
