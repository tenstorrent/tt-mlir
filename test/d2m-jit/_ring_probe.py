"""Milestone 1: unicast-to-neighbor probe for ring all_reduce.

Device p sends its tile to (p+1)%N and receives from (p-1)%N. After the round,
device q holds device (q-1)%N's tile. Validates the one unproven primitive the
ring needs: a single-neighbor fabric send with ring wraparound (3 -> 0).
"""
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _ring_send(in0, out0, start_sem, end_sem):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    # Global start barrier across the whole line (as in all_gather).
    device_synchronize(
        start_sem,
        start_device=[dy, 0],
        mcast_shape=[1, 4],
        num_receivers=3,
        core_indices=[cy, cx],
    )
    nbr = (p + 1) % 4  # ring +1 neighbor, wraps 3 -> 0
    t = remote_load(in0, [0, 0])
    # Unicast: send only to the neighbor device (device_mcast_shape=[1,1]).
    remote_store(
        out0,
        [0, 0],
        t,
        start_device=[dy, nbr],
        device_mcast_shape=[1, 1],
        semaphore=end_sem,
        semaphore_indices=[cy, 0],
    )
    semaphore_wait(end_sem, 1)  # exactly one send arrives (from p-1)


def main():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    L_out = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    full = torch.randn(32, N * 32, dtype=torch.float32)  # block d = X_d
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _ring_send(
        in_s, out_s, ss, es, grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="ring", routing="unidir_ring_torus"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, N])
    result = out_s.to_host()  # (32, N*32), block q = received tile

    print("=== ring unicast-to-neighbor probe (N=4) ===")
    for q in range(N):
        got = result[:, 32 * q : 32 * (q + 1)]
        exp = full[:, 32 * ((q - 1) % N) : 32 * ((q - 1) % N + 1)]  # X_{(q-1)%N}
        print(f"  device {q}: maxdiff vs X_{(q-1)%N} = {(got-exp).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
