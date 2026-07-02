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

Two-kernel port (all_gather then matmul), mirroring the reference's "use_non_fused"
path (all_gather_async then minimal_matmul). The weight is TP-sharded (the real
AGMM form), NOT replicated. Fusing the two kernels and async overlap are future
work (see tools/d2m-jit/agmm_design.md).

Scaling (vs. the original single-block port that capped at K=512 / N_out=512):
  - the matmul is a DISTRIBUTED TILED matmul -- M across grid rows, this device's
    N across grid columns, K split into nK sub-blocks reduced via NoC reads +
    accumulation -- so no single block/core overflows L1;
  - the TP weight is DISTRIBUTED across the [nK, gn] grid at shard time (a
    single-core weight residents the whole [K, N/N] slice and overflows L1 as K
    grows).
With these, K scales to 2560, N_out to 16384, M to 1024 on this 1x4 Blackhole.
The remaining wall to the reference's full sizes (K=5120, M=115200) is the
all_gather: it residents each device's full [MT,KT] shard on one fabric core
(<=2 fabric cores available), so MT*KT is L1-bounded. Lowering that via a
chunked-loop gather hits an intermittent in-loop fabric-ordering hang; the real
fix is the dedicated-fabric-worker CCL design / num_links>1. See agmm_design.md.
"""

import pytest
import torch
import d2m_jit as d2m
from utils import assert_pcc


def _largest_divisor_leq(x, cap):
    for d in range(min(x, cap), 0, -1):
        if x % d == 0:
            return d
    return 1


def _make_ag(N, MT, KT):
    # all_gather over the device ring: a single fabric core mcasts this device's
    # full [MT, KT] activation row to every device's gathered buffer.
    @d2m.kernel
    def ag(in0, out0, ss, es):
        dy = mesh_position(0)
        dx = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=N - 1, core_indices=[cy, cx])
        buf = empty([MT, KT])
        remote_load(buf, in0, [0, 0])               # this device's [MT,KT] shard
        remote_store(out0, [dx, 0], buf, start_device=[dy, 0],
                     device_mcast_shape=[1, N], semaphore=es, semaphore_indices=[cy, 0])
        semaphore_wait(es, N - 1)                    # gathered now complete everywhere
    return ag


def _make_mm(mb, nb, nK):
    # Distributed tiled matmul on a (gm,gn) core grid: core (cy,cx) accumulates
    # acc[mb,nb] += g[cy,k] @ w[k,cx] over nK K-sub-blocks (NoC reads). Only
    # [mb,KB]+[KB,nb] tiles are transient per step, so L1 stays small for large K.
    @d2m.kernel
    def mm(a, b, out):
        cy = core_index(0)
        cx = core_index(1)
        acc = zeros([mb, nb])
        for k in range(nK):
            acc += remote_load(a, [cy, k]) @ remote_load(b, [k, cx])
        remote_store(out, [cy, cx], acc)
    return mm


def _agmm(MT, KT, NTd, N):
    """M = N*MT*32 (gathered), K = KT*32, Nout = N*NTd*32 (TP-sharded /N)."""
    Mt = N * MT
    gm = _largest_divisor_leq(Mt, 10)         # M across grid rows (<=10)
    gn = _largest_divisor_leq(NTd, 8)         # this device's N across grid cols (<=8)
    cap = max(1, min(110 // gm, 110 // gn))    # all operand grids must be <= 110 cores
    nK = _largest_divisor_leq(KT, min(cap, 16))
    mb, nb, KB = Mt // gm, NTd // gn, KT // nK

    d2m.config.use_split_unified_thread_v2 = True
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(shape=(MT * 32, KT * 32), dtype=d2m.float32, block_shape=[MT, KT], grid_shape=[1, 1])
    L_g = d2m.Layout(shape=(N * MT * 32, KT * 32), dtype=d2m.float32, block_shape=[MT, KT], grid_shape=[N, 1])
    L_w = d2m.Layout(shape=(KT * 32, NTd * 32), dtype=d2m.float32, block_shape=[KB, nb], grid_shape=[nK, gn])
    L_o = d2m.Layout(shape=(N * MT * 32, NTd * 32), dtype=d2m.float32, block_shape=[mb, nb], grid_shape=[gm, gn])

    act = torch.randn(MT * 32, N * KT * 32, dtype=torch.float32)   # N sp shards (columns)
    W = torch.randn(KT * 32, N * NTd * 32, dtype=torch.float32)    # full weight (TP-sharded along N)

    in_s = d2m.reblock(d2m.mesh_shard(act, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    g_s = d2m.reblock(d2m.empty(L_g), [N, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make_ag(N, MT, KT)(in_s, g_s, ss, es, grid=(1, 1), fabric=d2m.fabric_config(cluster_axis=1))

    g_t = d2m.reblock(g_s, [gm, nK])                                          # block [mb, KB]
    w_t = d2m.reblock(d2m.mesh_shard(W, L_w, shard_dims=[0, 1], shard_shape=[1, N]), [nK, gn])
    o_s = d2m.reblock(d2m.empty(L_o), [gm, gn])                               # block [mb, nb]
    _make_mm(mb, nb, nK)(g_t, w_t, o_s, grid=(gm, gn))

    result = d2m.mesh_gather(o_s, shard_dims=[0, 1], shard_shape=[1, N]).to_host()
    gathered = torch.cat([act[:, p * KT * 32:(p + 1) * KT * 32] for p in range(N)], dim=0)
    expected = gathered @ W
    assert_pcc(expected, result)


# (MT_per_device, K_tiles, Nout_tiles_per_device). M = N*MT*32, N_out = N*NTd*32.
@pytest.mark.parametrize("MT,KT,NTd", [
    (1, 2, 2),     # M=128,  K=64,   N_out=256    (tiny baseline)
    (1, 8, 8),     # M=128,  K=256,  N_out=1024
    (2, 4, 4),     # M=256,  K=128,  N_out=512
    # K scaling (K up to the all_gather residency cap on this box):
    (1, 32, 8),    # K=1024
    (1, 80, 8),    # K=2560
    # N_out scaling (N is matmul-only / TP-sharded; the gather never touches it):
    (1, 8, 32),    # N_out=4096
    (1, 8, 128),   # N_out=16384
    # M scaling (gathered seq dim):
    (4, 8, 8),     # M=512
    (8, 8, 8),     # M=1024
])
def test_all_gather_matmul(MT, KT, NTd):
    _agmm(MT, KT, NTd, N=4)
