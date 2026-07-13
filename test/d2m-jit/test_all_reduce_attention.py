# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Larger ring all_reduce on realistic attention shapes from 30B-70B LLMs.

In tensor-parallel (TP=N) attention, each rank computes a partial attention
output (the o_proj result, shape [tokens, hidden]) and the ranks **all_reduce
(sum)** it so every rank holds the full output. Representative hidden sizes:

  - Llama-2/3 70B:  hidden = 8192  (= 256 tiles wide)
  - ~30-34B models: hidden ~ 6656-8192

The full [tokens, hidden] tensor is tiled across the Tensix core grid; each core
all_reduces its [M, K]-tile chunk. This test exercises that per-core chunk for
the runtime-loop ring all_reduce at the largest block that fits a single core's
L1 (~16 tiles with the per-iteration fabric buffers), which is 16x the 1-tile
shard of test_ring_all_reduce_loop. The kernel is generic over the per-core block
[M, K] and the TP degree N.

(Spreading one all_reduce across the whole core grid -- a multi-core fabric ring
covering the full 8192-wide hidden -- needs grid-level fabric support that the
ring kernel does not yet have; it compiles but fails at runtime today. So this
test stays single-core / per-core-chunk.)
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
        acc += remote_load(in0, [0, 0])         # this rank's partial (own shard)
        acc_prev = zeros([M, K])
        for k in range(N - 1):
            fwd = acc - acc_prev
            t = empty([M, K])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es,
                         semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)
            r = fabric_recv(t, [])
            acc_prev = copy_(acc_prev, acc)
            acc += r
        remote_store(out, [0, 0], acc)
    return _k


# Per-core attention all_reduce chunks (each == 16 tiles, the single-core max).
# (label, M_tiles, K_tiles) -> chunk = [M*32 tokens, K*32 hidden].
@pytest.mark.parametrize("label,M,K", [
    ("70B o_proj: 32 tok x 512 hidden chunk", 1, 16),
    ("70B o_proj: 128 tok x 128 hidden chunk", 4, 4),
    ("30B o_proj: 64 tok x 256 hidden chunk", 2, 8),
])
def test_attention_all_reduce_chunk(label, M, K):
    N = 4  # TP degree = mesh volume (this box is a 1x4 Blackhole)
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(M * 32, K * 32), dtype=d2m.float32,
                   block_shape=[M, K], grid_shape=[1, 1])
    # Distinct partial per TP rank: fi columns [p*K*32 : (p+1)*K*32] -> rank p.
    fi = torch.randn(M * 32, N * K * 32, dtype=torch.float32)
    ins = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    outs = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make(N, M, K)(ins, outs, ss, es, grid=(1, 1),
                   fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                            routing="unidir_ring_torus"))
    result = d2m.mesh_gather(outs, shard_dims=[0, 1], shard_shape=[1, N]).to_host()
    full = sum(fi[:, K * 32 * q:K * 32 * (q + 1)] for q in range(N))  # summed partials
    for p in range(N):
        assert_pcc(full, result[:, K * 32 * p:K * 32 * (p + 1)])
