"""Milestone 3a: single-step recv via a gridless #l1 tmp scratch (no NoC read-back).

PIVOT (2026-06-24): the recv buffer is now an in-kernel `empty([1,1])` -> gridless
#l1 scratch (the CCL tmp-buffer), NOT a #shard operand. Device p:
  - fabric-writes in0[0] into the NEIGHBOR's tmp scratch (cross-device
    remote_store with a scratch dst), then
  - waits, then s = fabric_recv(tmp) + in0[1]   (compute consumes the tmp CB
    view-free -- the scratch IS the shard, so no reinterpret_cast)
and stores s. Validates: (a) correctness of the symmetric-scratch fabric write,
(b) fabric_recv lowers with NO noc_async_read (reserve+push only).
"""
import sys
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _k(in0, out, start_sem, es):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(start_sem, start_device=[dy, 0], mcast_shape=[1, 4],
                       num_receivers=3, core_indices=[cy, cx])
    nbr = (p + 1) % 4
    tmp = empty([1, 1])
    a = remote_load(in0, [0, 0])
    remote_store(tmp, [], a, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es, semaphore_indices=[cy, 0])
    semaphore_wait(es, 1)
    s = fabric_recv(tmp, []) + remote_load(in0, [1, 0])
    remote_store(out, [0, 0], s)


def build():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(shape=(2 * 32, 32), dtype=d2m.float32, block_shape=[1, 1],
                      grid_shape=[2, 1])
    L1 = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                    grid_shape=[1, 1])
    fi = torch.randn(2 * 32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(fi, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [2, 1])
    out_s = d2m.reblock(d2m.empty(L1), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _k(in_s, out_s, ss, es, grid=(1, 1),
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
else:
    out, fi = build()
    r = out.to_host()
    print("=== milestone 3a: tmp-scratch fabric_recv 1-step ===")
    ok = True
    for p in range(N):
        got = r[:, 32 * p:32 * (p + 1)]
        exp = fi[0:32, 32 * ((p - 1) % N):32 * ((p - 1) % N + 1)] + fi[32:64, 32 * p:32 * (p + 1)]
        md = (got - exp).abs().max().item(); ok = ok and md < 0.05
        print(f"  device {p}: maxdiff={md:.4f}")
    print("RESULT:", "PASS" if ok else "FAIL")
