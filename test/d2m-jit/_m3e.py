"""Milestone 3e: GENERIC, LOOP-WRITTEN ring all_reduce for any mesh volume N.

Written with a `for k in static_range(N-1)` loop (unrolled at trace time) and
parameterized over the mesh volume N (a closed-over int capture), rather than
hand-unrolling for N=4. `static_range` is a DSL marker (see _src/ast.py) that
unrolls the loop body in the tracer -- so loop-carried accumulators stay plain
SSA reassignments (no scf.for iter_args -> no loop-carried-tensor bufferization
failure, no operand-store-in-loop re-alloc).

Algorithm: the circulate-and-accumulate ring (O(N) bandwidth -- sends the whole
shard each step). It keeps state in SSA (`acc`, `acc_prev`), avoiding the
operand read-modify-write that the bandwidth-optimal CHUNKED ring needs (which
requires runtime-indexed chunk slots -- not expressible as SSA, and reading a
grid operand back demotes it to #l1 where a local remote_store is rejected). So
the chunked variant stays the unrolled _m3d for now.

Per step k: forward `acc - acc_prev` (== the value received last step, a
send-only compute output), receive r, then acc += r. With acc_prev seeded to 0,
step 0 forwards the device's own shard. After N-1 steps every device holds the
full sum.
"""
import sys
import torch
import d2m_jit as d2m

N = 4


def make_kernel(N):
    # N is closed over -> captured as a compile-time int, so static_range(N-1)
    # and (p+1)%N are generic over the mesh volume. (device_synchronize's
    # num_receivers/mcast_shape are compile-time attrs that need literals; set
    # them to match the mesh.)
    @d2m.kernel
    def _k(in0, zin, out, ss, es):
        dy = mesh_position(0)
        p = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        # mcast_shape is generic ([1, N]); num_receivers must be a compile-time
        # literal (an i32 attribute) so it stays hardcoded to N-1 for this mesh.
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=3, core_indices=[cy, cx])
        nbr = (p + 1) % N
        v = remote_load(in0, [0, 0])
        acc = v
        # Seed acc_prev from an opaque zeros operand (NOT `v - v`, which folds to a
        # literal zero -> `acc - acc_prev` would canonicalize to `v` and send `v`
        # directly while compute also reads it: an illegal compute+DM CB share).
        acc_prev = remote_load(zin, [0, 0])
        for k in static_range(N - 1):
            t = empty([1, 1])
            remote_store(t, [], acc - acc_prev, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es, semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)
            r = fabric_recv(t, [])
            acc_prev = acc
            acc = acc + r
        remote_store(out, [0, 0], acc)
    return _k


def build():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[1, 1])
    fi = torch.randn(32, N * 32, dtype=torch.float32)
    zr = torch.zeros(32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    z_s = d2m.reblock(d2m.mesh_shard(zr, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    out_s = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _k = make_kernel(N)
    _k(in_s, z_s, out_s, ss, es, grid=(1, 1),
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
