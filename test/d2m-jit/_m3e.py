"""Milestone 3e: GENERIC, LOOP-WRITTEN ring all_reduce for any mesh volume N.

Written with a genuine `for k in range(N-1)` runtime `scf.for` loop (NOT the old
trace-time `static_range` unroll, which has been removed) and parameterized over
the mesh volume N. Loop-carried accumulators (`acc`, `acc_prev`) become scf.for
iter_args; the bufferization/operand-store issues that originally motivated the
unroll are resolved by the compute-initialized accumulator path (zeros-init +
`__add_acc__`, with the DM-seed guard rejecting `acc = remote_load(...)`).

Algorithm: the circulate-and-accumulate ring (O(N) bandwidth -- sends the whole
shard each step). Per step k: forward `acc - acc_prev` (== the value received
last step, a send-only compute output), receive r, then acc += r. With acc_prev
seeded to 0, step 0 forwards the device's own shard. After N-1 steps every device
holds the full sum.

Superseded as a test by `test_ring_all_reduce_loop.py`; kept as the milestone-3e
repro.
"""
import sys
import torch
import d2m_jit as d2m

N = 4


def make_kernel(N):
    # N is closed over -> generic over the mesh volume for the loop bound and
    # (p+1)%N. device_synchronize's num_receivers/mcast_shape are compile-time
    # attrs that need literals; set them to match the mesh.
    @d2m.kernel
    def _k(in0, out, ss, es):
        dy = mesh_position(0)
        p = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=3, core_indices=[cy, cx])
        nbr = (p + 1) % N
        acc = zeros([1, 1])
        own = empty([1, 1])
        remote_load(own, in0, [0, 0])
        acc += own
        acc_prev = zeros([1, 1])
        for k in range(N - 1):
            fwd = acc - acc_prev
            t = empty([1, 1])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es, semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)
            r = fabric_recv(t, [])
            acc_prev = copy_(acc_prev, acc)
            acc += r
        remote_store(out, [0, 0], acc)
    return _k


def build():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[1, 1])
    fi = torch.randn(32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    out_s = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _k = make_kernel(N)
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
    sys.exit(0)

out, fi = build()
r = out.to_host()
print(f"=== milestone 3e: GENERIC looped ring all_reduce (N={N}) ===")
full_S = sum(fi[:, 32 * q:32 * (q + 1)] for q in range(N))
ok = True
for p in range(N):
    got = r[:, 32 * p:32 * (p + 1)]
    md = (got - full_S).abs().max().item(); ok = ok and md < 0.05
    print(f"  device {p}: maxdiff={md:.4f}")
print("RESULT:", "PASS" if ok else "FAIL")
