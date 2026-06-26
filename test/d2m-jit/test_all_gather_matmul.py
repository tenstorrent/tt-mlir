# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""all_gather + matmul (the all_gather_minimal_matmul concept), d2m-jit port.

Ports the concept of TTNN's all_gather_minimal_matmul_async (Wan2.2 DiT linears)
to d2m-jit on a 1xN ring:

  - activations [M, K] are sequence-parallel sharded along M; each device has
    [M/N, K];
  - all_gather over the ring -> every device holds the full [M, K];
  - the weight [K, N_out] is tensor-parallel sharded along N_out; each device has
    [K, N_out/N];
  - matmul: gathered [M, K] @ weight_slice [K, N_out/N] -> [M, N_out/N] per
    device; mesh_gather concatenates the N-slices -> full [M, N_out].

It is a two-kernel port (all_gather then matmul), mirroring the reference's
"use_non_fused" path (all_gather_async then minimal_matmul). The weight is
TP-sharded (the real AGMM form), NOT replicated -- replicating an operand across
a multi-device mesh currently fails (buffer-size assert / compiler crash). Fusing
the two kernels and async overlap are future work (see tools/d2m-jit/agmm_design.md).
"""

import pytest
import torch
import d2m_jit as d2m
from utils import assert_pcc


def _make_ag(N, MT, KT):
    @d2m.kernel
    def ag(in0, out0, ss, es):
        dy = mesh_position(0)
        dx = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=N - 1, core_indices=[cy, cx])
        buf = empty([MT, KT])
        remote_load(buf, in0, [0, 0])               # this device's [MT,KT] activation shard
        remote_store(out0, [dx, 0], buf, start_device=[dy, 0],
                     device_mcast_shape=[1, N], semaphore=es, semaphore_indices=[cy, 0])
        semaphore_wait(es, N - 1)                    # gathered now complete on every device
    return ag


@d2m.kernel
def _mm(a, b, out):
    cy = core_index(0)                              # row-parallel: one gathered block per core
    remote_store(out, [cy, 0], remote_load(a, [cy, 0]) @ remote_load(b, [0, 0]))


# (MT_per_device, K_tiles, Nout_tiles_per_device). M = N*MT*32, N_out = N*NTd*32.
@pytest.mark.parametrize("MT,KT,NTd", [
    (1, 2, 2),    # M=128,  K=64,  N_out=256
    (1, 4, 4),    # M=128,  K=128, N_out=512
    (2, 4, 4),    # M=256,  K=128, N_out=512
    (1, 8, 8),    # M=128,  K=256, N_out=1024
])
def test_all_gather_matmul(MT, KT, NTd):
    N = 4  # ring / TP degree
    d2m.config.use_split_unified_thread_v2 = True
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(shape=(MT * 32, KT * 32), dtype=d2m.float32, block_shape=[MT, KT], grid_shape=[1, 1])
    L_g = d2m.Layout(shape=(N * MT * 32, KT * 32), dtype=d2m.float32, block_shape=[MT, KT], grid_shape=[N, 1])
    L_w = d2m.Layout(shape=(KT * 32, NTd * 32), dtype=d2m.float32, block_shape=[KT, NTd], grid_shape=[1, 1])
    L_o = d2m.Layout(shape=(N * MT * 32, NTd * 32), dtype=d2m.float32, block_shape=[MT, NTd], grid_shape=[N, 1])

    act = torch.randn(MT * 32, N * KT * 32, dtype=torch.float32)   # N sp shards (columns)
    W = torch.randn(KT * 32, N * NTd * 32, dtype=torch.float32)    # full weight (TP-sharded along N)

    in_s = d2m.reblock(d2m.mesh_shard(act, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    g_s = d2m.reblock(d2m.empty(L_g), [N, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make_ag(N, MT, KT)(in_s, g_s, ss, es, grid=(1, 1), fabric=d2m.fabric_config(cluster_axis=1))

    w_s = d2m.reblock(d2m.mesh_shard(W, L_w, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    o_s = d2m.reblock(d2m.empty(L_o), [N, 1])
    _mm(g_s, w_s, o_s, grid=(N, 1))

    result = d2m.mesh_gather(o_s, shard_dims=[0, 1], shard_shape=[1, N]).to_host()
    gathered = torch.cat([act[:, p * KT * 32:(p + 1) * KT * 32] for p in range(N)], dim=0)
    expected = gathered @ W
    assert_pcc(expected, result)
