"""Minimal repro: a fabric ring of >=3 interleaved cross-device sends HANGS on
device (1x4 Blackhole ring); the same kernel with 2 steps completes.

The headline finding is a clean boundary on the number of in-kernel READ-BACKS of
the output operand: **1 read-back tolerated, 2 deadlock.** (Not the send count.)

Each ring step here is:  fabric send -> semaphore_wait -> read back the OUTPUT
operand (the recv slot a peer fabric-wrote) -> add an input -> that result is the
next step's fabric send. Separate semaphore per step (es0/es1/es2), each waited
for value 1 (NOT a cumulative single semaphore). This 3-step kernel has TWO such
read-backs (steps 2 and 3) and hangs.

Verified elimination (all on the 1x4 ring, single fabric DM thread):
  - 2 or 3 fabric sends, forwarding INPUTS, NO read-back        -> completes
  - 2 fabric sends + compute add of INPUTS, no read-back        -> completes
  - 2 steps = 2 sends + ONE output read-back -> compute -> send -> completes
  - 3 steps = 3 sends + TWO output read-backs -> compute -> send -> HANGS

So it is NOT the send count (3 sends with no read-back complete), NOT the compute,
NOT the interleaved wait. The trigger is the SECOND `remote_load` read-back of the
output operand. That read-back lowers to a `noc_async_read` on NoC0 -- the same
NoC the fabric connection holds (ScheduleDMA pins multi-fabric-send to one DM
thread, processor=1/NoC0; the fabric connection is opened once / closed once, and
each step's codegen is identical, so this is a runtime resource conflict, not a
codegen bug). One output read-back interleaved with the open fabric connection on
NoC0 is fine; the second deadlocks (suspected NoC0 command-buffer / transaction-id
/ CB-depth contention between the read-back and the fabric connection).

Implication: the fix is to eliminate the DM read-back of the output operand --
model the cross-device fabric RECV as a compute-consumed input CB (cb_wait_front
on incoming, the tt-metal CCL pattern), or otherwise keep the recv read off the
fabric NoC. See tools/d2m-jit/all_reduce_design.md milestone 2b.

To get a passing 2-step version: drop the 3rd step block in `_ring_readback_accum`
(pass only es0/es1, grid_shape=[2,1], in0 with 2 rows).

Expected: completes. Actual (3 steps = 2 read-backs): HANGS (kill via timeout; the
device recovers cleanly afterward).

See tools/d2m-jit/all_reduce_design.md milestone 2b and the
`d2m-ring-interleaved-fabric-hang` memory.

Run (4-chip Blackhole box, full 1x4 mesh required -- a 1x2 sub-mesh cannot train
fabric here):
  python test/d2m-jit/repro_ring_fabric_readback_hang.py
"""
import torch
import d2m_jit as d2m

N = 4


@d2m.kernel
def _ring_readback_accum(in0, out, start_sem, es0, es1, es2):
    dy = mesh_position(0)
    p = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(start_sem, start_device=[dy, 0], mcast_shape=[1, 4],
                       num_receivers=3, core_indices=[cy, cx])
    nbr = (p + 1) % 4
    # Each step: fabric send -> wait -> read back the OUTPUT operand -> add an
    # input -> that becomes the next send's value. Separate semaphore per step
    # (es0/es1/es2), each waited for value 1 (NOT a cumulative single semaphore).
    # 2 such steps complete; 3 hang.
    s = remote_load(in0, [0, 0])
    remote_store(out, [0, 0], s, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es0, semaphore_indices=[cy, 0])
    semaphore_wait(es0, 1)
    s = remote_load(out, [0, 0]) + remote_load(in0, [1, 0])
    remote_store(out, [1, 0], s, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es1, semaphore_indices=[cy, 0])
    semaphore_wait(es1, 1)
    s = remote_load(out, [1, 0]) + remote_load(in0, [2, 0])
    remote_store(out, [2, 0], s, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                 semaphore=es2, semaphore_indices=[cy, 0])
    semaphore_wait(es2, 1)


def main():
    d2m.mesh((1, N), topology=("linear", "ring"))
    L_in = d2m.Layout(shape=(3 * 32, 32), dtype=d2m.float32, block_shape=[1, 1],
                      grid_shape=[3, 1])
    L_out = d2m.Layout(shape=(3 * 32, 32), dtype=d2m.float32, block_shape=[1, 1],
                       grid_shape=[3, 1])
    full = torch.randn(3 * 32, N * 32, dtype=torch.float32)
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, N]), [3, 1])
    out_s = d2m.reblock(d2m.empty(L_out), [3, 1])
    ss = d2m.global_semaphore()
    es0 = d2m.global_semaphore()
    es1 = d2m.global_semaphore()
    es2 = d2m.global_semaphore()
    _ring_readback_accum(
        in_s, out_s, ss, es0, es1, es2, grid=(1, 1),
        fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                 routing="unidir_ring_torus"))
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, N])
    result = out_s.to_host()  # reaching here means it did NOT hang
    print("=== completed (did NOT hang); shape", tuple(result.shape), "===")


if __name__ == "__main__":
    main()
