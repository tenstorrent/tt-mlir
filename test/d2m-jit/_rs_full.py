"""Milestone 2b: full ring reduce_scatter (N=4), separate end-sem per step.

Running-partial ring: device p ends owning the fully-reduced chunk (p+1)%N.
out grid [N,1]: slots 0..N-2 = per-step recv buffers, slot N-1 = result.
One end-semaphore per step (es0/es1/es2), each used once -> no cumulative count,
no reset race (single-semaphore cumulative hung; per-step reset segfaulted).
"""
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _reduce_scatter(in0, out, start_sem, es0, es1, es2):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start_sem, start_device=[dy, 0], mcast_shape=[1, 4],
        num_receivers=3, core_indices=[cy, cx],
    )
    nbr = (p + 1) % 4
    partial = remote_load(in0, [p, 0])  # c_p[p] = step-0 send value
    # step 0
    remote_store(out, [0, 0], partial, start_device=[dy, nbr],
                 device_mcast_shape=[1, 1], semaphore=es0, semaphore_indices=[cy, 0])
    semaphore_wait(es0, 1)
    partial = remote_load(out, [0, 0]) + remote_load(in0, [(p - 1) % 4, 0])
    # step 1
    remote_store(out, [1, 0], partial, start_device=[dy, nbr],
                 device_mcast_shape=[1, 1], semaphore=es1, semaphore_indices=[cy, 0])
    semaphore_wait(es1, 1)
    partial = remote_load(out, [1, 0]) + remote_load(in0, [(p - 2) % 4, 0])
    # step 2
    remote_store(out, [2, 0], partial, start_device=[dy, nbr],
                 device_mcast_shape=[1, 1], semaphore=es2, semaphore_indices=[cy, 0])
    semaphore_wait(es2, 1)
    partial = remote_load(out, [2, 0]) + remote_load(in0, [(p - 3) % 4, 0])
    # result -> slot N-1
    remote_store(out, [3, 0], partial)


def main():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(
        shape=(N * 32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[N, 1]
    )
    full_in = torch.randn(N * 32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(
        d2m.mesh_shard(full_in, L, shard_dims=[0, 1], shard_shape=[1, N]), [N, 1]
    )
    out_s = d2m.reblock(d2m.empty(L), [N, 1])
    ss = d2m.global_semaphore()
    es = [d2m.global_semaphore() for _ in range(3)]
    _reduce_scatter(
        in_s, out_s, ss, es[0], es[1], es[2], grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="ring", routing="unidir_ring_torus"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, N])
    result = out_s.to_host()

    print("=== ring reduce_scatter (N=4, separate sems) ===")
    ok = True
    for p in range(N):
        owned = (p + 1) % N
        got = result[32 * (N - 1):32 * N, 32 * p:32 * (p + 1)]
        exp = torch.zeros(32, 32)
        for e in range(N):
            exp += full_in[32 * owned:32 * owned + 32, 32 * e:32 * e + 32]
        md = (got - exp).abs().max().item()
        print(f"  device {p}: owns chunk {owned}, maxdiff = {md:.4f}")
        ok = ok and md < 0.05
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
