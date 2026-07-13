# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ring all_reduce written as a TRUE runtime `for k in range(N-1)` loop,
generic over the mesh volume N.

This is a genuine `scf.for` with fabric ops (remote_store + semaphore_wait +
fabric_recv) inside, and two loop-carried accumulators updated in place:

- `acc`     accumulates the received shards (`acc += r`, via __add_acc__);
- `acc_prev` snapshots the previous `acc` (`acc_prev = copy_(acc_prev, acc)`, the
  in-place copy that makes the snapshot bufferize as a loop iter_arg);
- each step forwards `acc - acc_prev` (== the value received last step, a fresh
  send-only compute output produced inside the loop, so its send CB is balanced).

The own shard enters via a PRE-loop `acc += remote_load(in0)`; that compute
intermediate is correctly materialized (the MarkSynchronizedBuffers fix). The
kernel is generic over N (a closed-over int): `range(N - 1)`, `mcast_shape=[1, N]`,
and `num_receivers=N - 1` (resolved at trace time via _eval_static_int). After
N-1 steps every device holds the full sum.
"""

import pytest
import torch
import d2m_jit as d2m
from utils import assert_pcc


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
        acc += remote_load(in0, [0, 0])        # own shard (compute-owned)
        acc_prev = zeros([1, 1])
        for k in range(N - 1):                  # runtime scf.for with fabric ops
            fwd = acc - acc_prev                # send-only forward (== last recv)
            t = empty([1, 1])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es,
                         semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)
            r = fabric_recv(t, [])
            acc_prev = copy_(acc_prev, acc)     # snapshot acc (in-place)
            acc += r                            # accumulate (in-place)
        remote_store(out, [0, 0], acc)
    return _k


# N is the mesh volume (1xN ring). The kernel is generic over N; this 4-chip
# Blackhole runs the full mesh (N=4) -- fabric needs the whole mesh, so smaller
# sub-meshes don't train here. On a larger box add more N to this list.
@pytest.mark.parametrize("N", [4])
def test_ring_all_reduce_runtime_loop(N):
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[1, 1])
    fi = torch.randn(32, N * 32, dtype=torch.float32)
    ins = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    outs = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make(N)(ins, outs, ss, es, grid=(1, 1),
             fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                      routing="unidir_ring_torus"))
    result = d2m.mesh_gather(outs, shard_dims=[0, 1], shard_shape=[1, N]).to_host()
    full = sum(fi[:, 32 * q:32 * (q + 1)] for q in range(N))
    for p in range(N):
        assert_pcc(full, result[:, 32 * p:32 * (p + 1)])
