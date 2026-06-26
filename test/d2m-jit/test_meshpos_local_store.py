# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression: mesh_position used in a thread that does no cross-device fabric op.

`mesh_position` lowers to `get_my_logical_mesh_position(fcm, dim)` and so needs a
fabric connection manager to dominate it (D2MToTTKernel). The FCM was only created
in funcs that contain a *fabric op* (cross-device store / semaphore). But a thread
can use mesh_position without itself doing a fabric op -- e.g. a circulate ring
where the LOCAL output store's grid index is `(mesh_position(1) - 1 - k) % N`. After
split-v2 that local store (and its mesh_position) lands on a different NoC thread
than the fabric send, so its func had no fabric op -> no FCM -> mesh_position
asserted "Expected fabric connection manager op" (D2MToTTKernel.cpp:174, confirmed
via a Debug build backtrace through D2MMeshPositionRewriter).

Fix: create the FCM when a func has any *fcm user* (fabric op OR mesh_position),
not just a fabric op. The setup only opens `num_send_dir` connections (0 for a
non-sending thread), so it is just the cheap topology build mesh_position needs.

Lower-only: the fix is a compile-time crash fix.
"""

import pytest
import torch
import d2m_jit as d2m
from d2m_jit._src.builder import (_Builder, _build_pipeline,
                                  _emit_returns_and_finalise,
                                  _get_system_desc_path)
from ttmlir.passmanager import PassManager


def _make(N, MT, KT):
    @d2m.kernel
    def _k(in0, out, ss, es):
        dy = mesh_position(0)
        p = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=N - 1, core_indices=[cy, cx])
        nbr = (p + 1) % N
        cur = zeros([MT, KT])
        cur += remote_load(in0, [0, 0])
        remote_store(out, [p, 0], cur)                  # local store, mesh-pos index
        for k in range(N - 1):
            fwd = empty([MT, KT])
            fwd = copy_(fwd, cur)
            t = empty([MT, KT])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es,
                         semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)
            cur = copy_(cur, fabric_recv(t, []))
            remote_store(out, [(p - 1 - k) % N, 0], cur)  # local store on a NoC
            #                                               thread w/o a fabric op
    return _k


def test_meshpos_local_store_lowers():
    sd = _get_system_desc_path()
    if not sd:
        pytest.skip("no system descriptor available")
    N, MT, KT = 4, 1, 2
    d2m.config.use_split_unified_thread_v2 = True
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(shape=(MT * 32, KT * 32), dtype=d2m.float32,
                      block_shape=[MT, KT], grid_shape=[1, 1])
    L_o = d2m.Layout(shape=(N * MT * 32, KT * 32), dtype=d2m.float32,
                     block_shape=[MT, KT], grid_shape=[N, 1])
    act = torch.randn(MT * 32, N * KT * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(act, L_in, shard_dims=[0, 1],
                                      shard_shape=[1, N]), [1, 1])
    o_s = d2m.reblock(d2m.empty(L_o), [N, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make(N, MT, KT)(in_s, o_s, ss, es, grid=(1, 1),
                     fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                              routing="unidir_ring_torus"))
    out = d2m.mesh_gather(o_s, shard_dims=[0, 1], shard_shape=[1, N])

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
