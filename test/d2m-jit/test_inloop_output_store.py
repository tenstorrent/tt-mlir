# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression: a `remote_store` to an OUTPUT operand INSIDE a runtime scf.for
must COMPILE (no longer crash `d2m-allocate`).

Previously this segfaulted `d2m-allocate`: the aliased-store copy-elision check
did `isa<OperandAliasOp>(remoteStoreOp.getLocalBuffer().getDefiningOp())`, but for
an in-loop store the local buffer is a loop-carried scf.for iter_arg (a block
argument) whose `getDefiningOp()` is null, so `isa<>` dereferenced null. The
store-once-after-the-loop form (test_ring_all_reduce_loop) stores the scf.for
*result* (has a defining op), so it never hit the null. Fixed by using
`isa_and_nonnull` (Allocate.cpp); the block-arg case then falls through to the
existing loop-carried handling that keeps the explicit store.

Lower-only (build -> full pipeline, no execution): the fix is a compile-time
crash fix. The kernel is a circulate ring whose accumulator is stored to the
output every iteration (in-loop), which is the pattern an incremental-output CCL
kernel (e.g. a fused all_gather+matmul emitting one row-block per step) needs.
"""

import pytest
import torch
import d2m_jit as d2m
from d2m_jit._src.builder import (_Builder, _build_pipeline,
                                  _emit_returns_and_finalise,
                                  _get_system_desc_path)
from ttmlir.passmanager import PassManager


def _make(N):
    @d2m.kernel
    def _k(in0, out, ss, es):
        dy = mesh_position(0)
        p = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=N - 1, core_indices=[cy, cx])
        nbr = (p + 1) % N
        acc = zeros([1, 1])
        acc += remote_load(in0, [0, 0])
        acc_prev = zeros([1, 1])
        for k in range(N - 1):
            fwd = acc - acc_prev
            t = empty([1, 1])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es,
                         semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)
            r = fabric_recv(t, [])
            acc_prev = copy_(acc_prev, acc)
            acc += r
            remote_store(out, [0, 0], acc)     # <-- IN-LOOP output store (the fix)
    return _k


def test_inloop_output_store_lowers():
    sd = _get_system_desc_path()
    if not sd:
        pytest.skip("no system descriptor available")
    N = 4
    d2m.config.use_split_unified_thread_v2 = True
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[1, 1])
    ins = d2m.reblock(d2m.mesh_shard(torch.randn(32, N * 32), L,
                                     shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    outs = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make(N)(ins, outs, ss, es, grid=(1, 1),
             fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                      routing="unidir_ring_torus"))
    out = d2m.mesh_gather(outs, shard_dims=[0, 1], shard_shape=[1, N])

    # Build + lower through the full pipeline (no execution): this is what used
    # to segfault in d2m-allocate.
    b = _Builder.get()
    _emit_returns_and_finalise(b, [out._resolve()])
    PassManager.parse(
        f"builtin.module(ttcore-register-device{{system-desc-path={sd} "
        f"mesh-shape=1,{N} mesh-topology=linear,ring}})",
        context=b.ctx,
    ).run(b.module.operation)
    PassManager.parse(f"builtin.module({_build_pipeline()})",
                      context=b.ctx).run(b.module.operation)
    b.module.operation.verify()
