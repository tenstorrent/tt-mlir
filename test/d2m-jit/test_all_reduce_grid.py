# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-core (grid) ring all_reduce.

The runtime-loop ring all_reduce, distributed across a Tensix core grid: each
core (cy, cx) all_reduces its [M, K]-tile sub-block of the per-device shard,
exchanging with the SAME core on neighbour devices over fabric.

This needs two things beyond the single-core ring:
  - num_links in fabric_config raised so enough fabric routing planes exist:
    cores <= num_links * cores_per_link (cores_per_link = 2 for unidir_ring_torus).
    The box has only 2 ethernet channels between adjacent devices, so num_links <= 2
    -> at most 4 fabric cores (a 2x2 grid).
  - the D2MToTTKernel fix so the cross-device fabric write into the gridless #l1
    scratch targets THIS core on the peer (my_logical coords), not a hardcoded
    (0,0) -- otherwise every core's data collides on the peer's core (0,0).

Combined with a multi-tile per-core block this reaches 4 cores x up to ~16 tiles
= up to 64 tiles per device in one pass (e.g. a [256, 256] or [64, 1024] chunk of
a TP attention all_reduce). Scaling past the 4-fabric-core / 2-eth-channel ceiling
needs a dedicated-fabric-core CCL design (the all_gather/matmul-async approach),
not every-core-does-fabric.
"""

import pytest
import torch
import d2m_jit as d2m
from utils import assert_pcc


def _make(N, M, K):
    @d2m.kernel
    def _k(in0, out, ss, es):
        dy = mesh_position(0)
        p = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=N - 1, core_indices=[cy, cx])
        nbr = (p + 1) % N
        acc = zeros([M, K])
        own = empty([M, K])
        remote_load(own, in0, [cy, cx])          # this core's sub-block
        acc += own
        acc_prev = zeros([M, K])
        for k in range(N - 1):
            fwd = acc - acc_prev
            t = empty([M, K])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es,
                         semaphore_indices=[cy, cx])
            semaphore_wait(es, k + 1)
            r = fabric_recv(t, [])
            acc_prev = copy_(acc_prev, acc)
            acc += r
        remote_store(out, [cy, cx], acc)
    return _k


# (label, GY, GX, M, K, num_links). cores = GY*GX must be <= num_links*2 (<= 4 here).
@pytest.mark.parametrize("label,GY,GX,M,K,num_links", [
    ("2x1 grid, [2,8]/core (128 tok x 256 hid)", 2, 1, 2, 8, 1),
    ("2x2 grid, [1,16]/core (64 tok x 1024 hid)", 2, 2, 1, 16, 2),
    ("2x2 grid, [4,4]/core (256 tok x 256 hid)", 2, 2, 4, 4, 2),
])
def test_grid_all_reduce(label, GY, GX, M, K, num_links):
    N = 4  # TP degree = mesh volume
    d2m.config.use_split_unified_thread_v2 = True
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(GY * M * 32, GX * K * 32), dtype=d2m.float32,
                   block_shape=[M, K], grid_shape=[GY, GX])
    fi = torch.randn(GY * M * 32, N * GX * K * 32, dtype=torch.float32)
    ins = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [GY, GX])
    outs = d2m.reblock(d2m.empty(L), [GY, GX])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make(N, M, K)(ins, outs, ss, es, grid=(GY, GX),
                   fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                            routing="unidir_ring_torus",
                                            num_links=num_links))
    result = d2m.mesh_gather(outs, shard_dims=[0, 1], shard_shape=[1, N]).to_host()
    W = GX * K * 32
    full = sum(fi[:, W * q:W * (q + 1)] for q in range(N))  # summed TP partials
    for p in range(N):
        assert_pcc(full, result[:, W * p:W * (p + 1)])
