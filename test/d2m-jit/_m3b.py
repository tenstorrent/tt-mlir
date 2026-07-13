"""Milestone 3b: 2-step chained ring via tmp #l1 scratches (the case that HUNG).

Each device runs two chained send->wait->recv->accumulate steps; step t+1's send
payload depends on step t's recv+compute. This is exactly the structure that
deadlocked before (the 2nd output read-back on NoC0 contending with the open
fabric connection). With the tmp-scratch fabric_recv (view-free, no noc_async_read)
it should be hang-free.

Running sum over the ring (N=4):
  acc = in0[0]
  step0: send acc -> nbr; wait; acc = recv + in0[1]
  step1: send acc -> nbr; wait; acc = recv + in0[2]
  store acc
=> device p final = in0[0]_{p-2} + in0[1]_{p-1} + in0[2]_p   (indices mod N)
"""
import sys
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _k(in0, out, start_sem, es0, es1):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(start_sem, start_device=[dy, 0], mcast_shape=[1, 4],
                       num_receivers=3, core_indices=[cy, cx])
    nbr = (p + 1) % 4
    tmp0 = empty([1, 1])
    tmp1 = empty([1, 1])
    # step 0
    acc0 = remote_load(in0, [0, 0])
    remote_store(tmp0, [], acc0, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es0, semaphore_indices=[cy, 0])
    semaphore_wait(es0, 1)
    acc1 = fabric_recv(tmp0, []) + remote_load(in0, [1, 0])
    # step 1
    remote_store(tmp1, [], acc1, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es1, semaphore_indices=[cy, 0])
    semaphore_wait(es1, 1)
    acc2 = fabric_recv(tmp1, []) + remote_load(in0, [2, 0])
    remote_store(out, [0, 0], acc2)


def build():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(shape=(3 * 32, 32), dtype=d2m.float32, block_shape=[1, 1],
                      grid_shape=[3, 1])
    L1 = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                    grid_shape=[1, 1])
    fi = torch.randn(3 * 32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(fi, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [3, 1])
    out_s = d2m.reblock(d2m.empty(L1), [1, 1])
    ss = d2m.global_semaphore()
    es0 = d2m.global_semaphore()
    es1 = d2m.global_semaphore()
    _k(in_s, out_s, ss, es0, es1, grid=(1, 1),
       fabric=d2m.fabric_config(cluster_axis=1, topology="ring", routing="unidir_ring_torus"))
    out = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, N])
    return out, fi


if len(sys.argv) > 1 and sys.argv[1] == "dump":
    from d2m_jit._src.builder import (_Builder, _build_pipeline,
                                      _emit_returns_and_finalise, _get_system_desc_path)
    from ttmlir.passmanager import PassManager
    out, fi = build()
    b = _Builder.get(); _emit_returns_and_finalise(b, [out._resolve()])
    sd = _get_system_desc_path()
    PassManager.parse(f"builtin.module(ttcore-register-device{{system-desc-path={sd} mesh-shape=1,4 mesh-topology=linear,linear}})", b.ctx).run(b.module.operation)
    full = _build_pipeline().split(",")
    stop = next(i for i, p in enumerate(full) if p.startswith("d2m-to-ttkernel"))
    PassManager.parse(f"builtin.module({','.join(full[:stop+1])})", b.ctx).run(b.module.operation)
    print(b.module)
    sys.exit(0)

out, fi = build()
r = out.to_host()
print("=== milestone 3b: 2-step chained ring (tmp scratch) ===")
ok = True
for p in range(N):
    got = r[:, 32 * p:32 * (p + 1)]
    exp = (fi[0:32, 32 * ((p - 2) % N):32 * ((p - 2) % N + 1)]
           + fi[32:64, 32 * ((p - 1) % N):32 * ((p - 1) % N + 1)]
           + fi[64:96, 32 * p:32 * (p + 1)])
    md = (got - exp).abs().max().item(); ok = ok and md < 0.05
    print(f"  device {p}: maxdiff={md:.4f}")
print("RESULT:", "PASS" if ok else "FAIL")
