"""Milestone 3c: full ring all_reduce (N-1 chained steps, tmp #l1 scratch).

Circulate-and-accumulate ring all_reduce (N=4, 3 steps). Each device forwards
what it received last step and accumulates the received value:
  v   = my shard
  acc = v ; fwd = v
  step k (1..3): send fwd -> nbr; wait; r = recv; acc += r; fwd = r
=> acc = v_p + v_{p-1} + v_{p-2} + v_{p-3} = sum over ALL devices.
Every device ends with the same full sum -- a true all_reduce.

(Not bandwidth-optimal -- sends the whole shard each step rather than 1/N chunks
-- but it exercises the full (N-1)-step chained ring on the tmp-scratch fabric_recv
primitive and is trivially verifiable.)
"""
import sys
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _k(in0, out, start_sem, es1, es2, es3):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(start_sem, start_device=[dy, 0], mcast_shape=[1, 4],
                       num_receivers=3, core_indices=[cy, cx])
    nbr = (p + 1) % 4
    t1 = empty([1, 1])
    t2 = empty([1, 1])
    t3 = empty([1, 1])
    # A value that is fabric-sent must be send-only (a CB consumed by BOTH a DM
    # send and compute does not legalize), so we never forward a raw recv result
    # nor a compute-consumed load directly. Instead: load in0[0] twice (a
    # send-only copy + a compute copy), and forward each received value as a
    # send-only compute output  acc_k - acc_{k-1}  (== r_k), keeping the raw
    # recv result single-(compute-)consumer.
    v_send = remote_load(in0, [0, 0])
    v = remote_load(in0, [0, 0])
    # step 1: forward my own shard (send-only copy)
    remote_store(t1, [], v_send, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es1, semaphore_indices=[cy, 0])
    semaphore_wait(es1, 1)
    r1 = fabric_recv(t1, [])
    acc1 = v + r1
    # step 2: forward r1 == acc1 - v  (send-only compute output)
    remote_store(t2, [], acc1 - v, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es2, semaphore_indices=[cy, 0])
    semaphore_wait(es2, 1)
    r2 = fabric_recv(t2, [])
    acc2 = acc1 + r2
    # step 3: forward r2 == acc2 - acc1  (send-only compute output)
    remote_store(t3, [], acc2 - acc1, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es3, semaphore_indices=[cy, 0])
    semaphore_wait(es3, 1)
    r3 = fabric_recv(t3, [])
    acc3 = acc2 + r3
    remote_store(out, [0, 0], acc3)


def build():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[1, 1])
    fi = torch.randn(32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    out_s = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es1 = d2m.global_semaphore()
    es2 = d2m.global_semaphore()
    es3 = d2m.global_semaphore()
    _k(in_s, out_s, ss, es1, es2, es3, grid=(1, 1),
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
print("=== milestone 3c: ring all_reduce (3 steps) ===")
full_sum = sum(fi[:, 32 * q:32 * (q + 1)] for q in range(N))
ok = True
for p in range(N):
    got = r[:, 32 * p:32 * (p + 1)]
    md = (got - full_sum).abs().max().item(); ok = ok and md < 0.05
    print(f"  device {p}: maxdiff={md:.4f}")
print("RESULT:", "PASS" if ok else "FAIL")
