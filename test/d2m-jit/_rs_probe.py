"""Milestone 2a: in-kernel read-back of a neighbor-pushed operand.

The reduce_scatter ring must READ (in-kernel) the tile a neighbor pushed into a
shared operand, to accumulate it. No existing kernel reads an operand it also
writes. This probes exactly that: device p pushes its tile into (p+1)'s `buf`
slot, waits, then remote_loads its OWN `buf` slot and stores it to `out`.
If out[p] == in[(p-1)%4], read+write of one cross-device operand is expressible.
"""
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _ring_readback(in0, out, start_sem, end_sem):
    # Single output `out` with 2 tile slots (grid [2,1]): slot 0 = recv buffer
    # (neighbor pushes here), slot 1 = result (local write). Probes read+write of
    # the one output operand in-kernel.
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start_sem,
        start_device=[dy, 0],
        mcast_shape=[1, 4],
        num_receivers=3,
        core_indices=[cy, cx],
    )
    nbr = (p + 1) % 4
    t = remote_load(in0, [0, 0])
    remote_store(
        out,
        [0, 0],
        t,
        start_device=[dy, nbr],
        device_mcast_shape=[1, 1],
        semaphore=end_sem,
        semaphore_indices=[cy, 0],
    )
    semaphore_wait(end_sem, 1)
    r = remote_load(out, [0, 0])  # read back what (p-1) pushed into my slot 0
    remote_store(out, [1, 0], r)  # local store to slot 1


def main():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    L_out = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    full = torch.randn(32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [2, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _ring_readback(
        in_s, out_s, ss, es, grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="ring", routing="unidir_ring_torus"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, N])
    result = out_s.to_host()  # (64, N*32); device q block, rows 32:64 = slot 1
    print("=== ring in-kernel read-back probe (N=4) ===")
    for q in range(N):
        got = result[32:64, 32 * q : 32 * (q + 1)]  # slot 1 of device q
        exp = full[:, 32 * ((q - 1) % N) : 32 * ((q - 1) % N + 1)]
        print(f"  device {q}: maxdiff vs X_{(q-1)%N} = {(got-exp).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
