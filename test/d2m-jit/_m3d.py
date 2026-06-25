"""Milestone 3d: bandwidth-optimal CHUNKED ring all_reduce (reduce-scatter + all-gather).

Each device holds a vector of N chunks; every device ends with the full sum S of
all devices' vectors (a true all_reduce). Unlike _m3c (sends the whole vector each
step, O(N) data), this moves only 1 chunk per step (O(1)) -- the standard ring
reduce-scatter then all-gather, N-1 steps each.

Chunk indices in the ring depend on the device id p (runtime). We track chunks in
p-RELATIVE order so the per-step logic uses CONSTANT rel-indices (unrollable in
Python); only the operand load/store use the runtime actual index (p+r)%N.

Reduce-scatter (rel: send -k, accumulate into -k-1):
  step0: send c[0]; c[3] += recv     step1: send c[3]; c[2] += recv
  step2: send c[2]; c[1] += recv   => c[1] (actual p+1) is fully reduced.
All-gather (rel: send 1-k, place into -k); place == add to a zero tile so the
result is a send-only compute output:
  step0: send c[1]; c[0] = 0+recv   step1: send c[0]; c[3] = 0+recv
  step2: send c[3]; c[2] = 0+recv  => all chunks reduced.
"""
import sys
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _k(in0, zin, out, ss, e0, e1, e2, e3, e4, e5):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, 4],
                       num_receivers=3, core_indices=[cy, cx])
    nbr = (p + 1) % 4

    c0 = remote_load(in0, [(p + 0) % 4, 0])  # rel 0 (actual p): RS step0 send-only
    c1 = remote_load(in0, [(p + 1) % 4, 0])
    c2 = remote_load(in0, [(p + 2) % 4, 0])
    c3 = remote_load(in0, [(p + 3) % 4, 0])
    zt = remote_load(zin, [0, 0])            # zero tile for all-gather placement
    t0 = empty([1, 1]); t1 = empty([1, 1]); t2 = empty([1, 1])
    t3 = empty([1, 1]); t4 = empty([1, 1]); t5 = empty([1, 1])

    # ---- reduce-scatter ----
    remote_store(t0, [], c0, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=e0, semaphore_indices=[cy, 0])
    semaphore_wait(e0, 1)
    c3 = c3 + fabric_recv(t0, [])
    remote_store(t1, [], c3, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=e1, semaphore_indices=[cy, 0])
    semaphore_wait(e1, 1)
    c2 = c2 + fabric_recv(t1, [])
    remote_store(t2, [], c2, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=e2, semaphore_indices=[cy, 0])
    semaphore_wait(e2, 1)
    c1 = c1 + fabric_recv(t2, [])
    # c1 is now the fully-reduced chunk this device owns (actual (p+1)%4).

    # ---- all-gather ----
    # A reduced chunk must be both FORWARDED (fabric send) and kept for the final
    # local store. A value cannot be consumed by two DM ops (the first pops the
    # CB, the second's wait_front hangs), so for each chunk make TWO independent
    # send-only/store-only compute copies (add-to-zero); the raw recv stays a
    # compute consumer only.
    out1 = zt + c1                       # store copy of owned reduced chunk (rel 1)
    remote_store(t3, [], zt + c1, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=e3, semaphore_indices=[cy, 0])
    semaphore_wait(e3, 1)
    r0 = fabric_recv(t3, [])
    out0 = zt + r0                       # rel 0 reduced
    remote_store(t4, [], zt + r0, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=e4, semaphore_indices=[cy, 0])
    semaphore_wait(e4, 1)
    r3 = fabric_recv(t4, [])
    out3 = zt + r3                       # rel 3 reduced
    remote_store(t5, [], zt + r3, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=e5, semaphore_indices=[cy, 0])
    semaphore_wait(e5, 1)
    out2 = zt + fabric_recv(t5, [])      # rel 2 reduced (last; store only)

    remote_store(out, [(p + 0) % 4, 0], out0)
    remote_store(out, [(p + 1) % 4, 0], out1)
    remote_store(out, [(p + 2) % 4, 0], out2)
    remote_store(out, [(p + 3) % 4, 0], out3)


def build():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(N * 32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[N, 1])
    Lz = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                    grid_shape=[1, 1])
    fi = torch.randn(N * 32, N * 32, dtype=torch.float32)
    zr = torch.zeros(32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [N, 1])
    z_s = d2m.reblock(d2m.mesh_shard(zr, Lz, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    out_s = d2m.reblock(d2m.empty(L), [N, 1])
    ss = d2m.global_semaphore()
    es = [d2m.global_semaphore() for _ in range(6)]
    _k(in_s, z_s, out_s, ss, *es, grid=(1, 1),
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
print("=== milestone 3d: chunked ring all_reduce (reduce-scatter + all-gather) ===")
full_S = sum(fi[:, 32 * q:32 * (q + 1)] for q in range(N))  # (N*32, 32)
ok = True
for p in range(N):
    got = r[:, 32 * p:32 * (p + 1)]
    md = (got - full_S).abs().max().item(); ok = ok and md < 0.05
    print(f"  device {p}: maxdiff={md:.4f}")
print("RESULT:", "PASS" if ok else "FAIL")
