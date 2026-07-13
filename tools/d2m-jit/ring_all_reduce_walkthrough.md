# Runtime-loop ring all_reduce — kernel + test walkthrough

A single readable place to review the runtime-loop ring all_reduce: the d2m-jit
DSL kernel, the pytest, and the actual tt-metal kernels the compiler lowers it to.

## Where it lives

| What | Path |
| --- | --- |
| Single-core kernel + test (shipped) | `test/d2m-jit/test_ring_all_reduce_loop.py` |
| Multi-core (grid) kernel + test (shipped) | `test/d2m-jit/test_all_reduce_grid.py` |
| Milestone-3e repro (standalone, has a `dump` mode) | `test/d2m-jit/_m3e.py` |
| DSL support (`__add_acc__`, `copy_`, `_eltwise_block`) | `tools/d2m-jit/api.py` |
| DSL support (`visit_For` scf.for, `visit_AugAssign`, DM-seed guard) | `tools/d2m-jit/_src/ast.py` |
| Design log + root-cause history | `tools/d2m-jit/all_reduce_design.md` |

Run it:
```bash
source env/activate
export PYTHONPATH=$PWD/build-d2m-jit/python_packages:$PWD/build-d2m-jit/runtime/python
export SYSTEM_DESC_PATH=$PWD/ttrt-artifacts/system_desc.ttsys
tt-smi -r all                        # reset between fabric runs
cd test/d2m-jit && python -m pytest test_ring_all_reduce_loop.py -q
# or the standalone repro, including an IR dump:
python _m3e.py            # runs on device, prints PASS/FAIL
python _m3e.py dump       # prints the lowered ttkernel IR (shown below)
```

## The kernel (d2m-jit DSL)

Circulate-and-accumulate ring: each of N devices holds a `[32,32]` shard; after
`N-1` fabric steps every device holds the full sum. The loop is a genuine runtime
`scf.for` with fabric ops inside.

```python
def _make(N):
    @d2m.kernel
    def _k(in0, out, ss, es):
        dy = mesh_position(0)
        p  = mesh_position(1)             # this device's ring position
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                           num_receivers=N - 1, core_indices=[cy, cx])
        nbr = (p + 1) % N                 # forward neighbour on the ring
        acc = zeros([1, 1])              # COMPUTE-owned init (key! not DM-seeded)
        acc += remote_load(in0, [0, 0])  # fold in own shard (pre-loop)
        acc_prev = zeros([1, 1])
        for k in range(N - 1):            # runtime scf.for, fabric ops inside
            fwd = acc - acc_prev          # send-only forward (== last recv)
            t = empty([1, 1])
            remote_store(t, [], fwd, start_device=[dy, nbr],
                         device_mcast_shape=[1, 1], semaphore=es,
                         semaphore_indices=[cy, 0])
            semaphore_wait(es, k + 1)     # cumulative count: wait for step k's recv
            r = fabric_recv(t, [])
            acc_prev = copy_(acc_prev, acc)  # snapshot acc (in-place -> iter_arg)
            acc += r                         # accumulate received shard (in-place)
        remote_store(out, [0, 0], acc)
    return _k
```

Three things make the runtime loop work (each was a separate wall, see the design
log):

1. **The accumulator init must be compute-owned.** `acc = zeros(...)` then
   `acc += ...` keeps the init and the compute pack on the *same* CB page. A
   DM-seeded init (`acc = remote_load(...)`) puts them on different FIFO pages and
   silently miscomputes — so the DSL now *rejects* that pattern at trace time
   (the DM-seed guard in `ast.py`).
2. **`acc += x` lowers via `__add_acc__`** (an in-place eltwise accumulate, the
   eltwise dual of matmul's `c += a@b`), so the loop-carried `acc` bufferizes as
   an `scf.for` iter_arg instead of failing one-shot-bufferize.
3. **`acc_prev = copy_(acc_prev, acc)`** is an in-place copy (`src + 0` written to
   `outs(dst)`); a plain identity copy gets canonicalized away and breaks the
   iter_arg aliasing.

## The test

```python
@pytest.mark.parametrize("N", [4])     # 4-chip Blackhole runs the full mesh
def test_ring_all_reduce_runtime_loop(N):
    d2m.mesh((1, N), topology=("linear", "ring"))
    L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1],
                   grid_shape=[1, 1])
    fi = torch.randn(32, N * 32, dtype=torch.float32)
    ins  = d2m.reblock(d2m.mesh_shard(fi, L, shard_dims=[0, 1], shard_shape=[1, N]), [1, 1])
    outs = d2m.reblock(d2m.empty(L), [1, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _make(N)(ins, outs, ss, es, grid=(1, 1),
             fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                      routing="unidir_ring_torus"))
    result = d2m.mesh_gather(outs, shard_dims=[0, 1], shard_shape=[1, N]).to_host()
    full = sum(fi[:, 32 * q:32 * (q + 1)] for q in range(N))  # reference sum
    for p in range(N):
        assert_pcc(full, result[:, 32 * p:32 * (p + 1)])      # every device == full sum
```

## What it lowers to (the actual tt-metal kernel)

`python _m3e.py dump`. The fabric `d2m.generic` becomes three threads: a compute
kernel and two NoC data-movement kernels. `arg_spec` attributes are elided for
readability.

### Compute kernel — the loop-carried accumulate

Note the structure: a pre-loop section seeds `acc` and `acc_prev` with `fill_tile`
(the `zeros`) and folds in the own shard, then the `scf.for %arg0 = 0 to 3`
(== `N-1`) body does, per step: produce `fwd = acc - acc_prev` (`sub_binary_tile`,
packed to the send CB), wait the semaphore, then `acc_prev = acc` and `acc += r`
(`add_binary_tile`). Every accumulate is `copy_tile` + binary-tile +
`pack_tile<true>` (out-of-order pack, pinned in place), guarded by
`unpack_stall_on_pack`.

```mlir
func.func private @compute_kernel5() {
  // --- pre-loop: acc = zeros; acc += own shard; acc_prev = zeros ---
  ttkernel.tile_regs_acquire()
  ttkernel.fill_tile(%c0, %cst)                 // acc = 0  (zeros init)
  ttkernel.pack_tile(%c0, %3, %c0, true)        // -> acc CB
  ttkernel.tile_regs_release()
  ttkernel.cb_wait_front(%7, %c1_i32)           // own shard (remote_load)
  ttkernel.experimental::unpack_stall_on_pack
  ttkernel.tile_regs_acquire()
  ttkernel.copy_tile(%3, %c0, %c0)              // acc
  ttkernel.copy_tile(%7, %c0, %c1)              // own
  ttkernel.add_binary_tile(%c0, %c1, %c0)       // acc += own
  ttkernel.pack_tile(%c0, %5, %c0, true)
  ttkernel.tile_regs_release()
  ttkernel.fill_tile(%c0, %cst)                 // acc_prev = 0
  ttkernel.pack_tile(%c0, %2, %c0, true)

  scf.for %arg0 = %c0 to %c3 step %c1 {         // for k in range(N-1)
    // fwd = acc - acc_prev  (send-only output CB %6)
    ttkernel.cb_reserve_back(%6, %c1_i32)
    ttkernel.experimental::unpack_stall_on_pack
    ttkernel.copy_tile(%5, %c0, %c0)            // acc
    ttkernel.copy_tile(%2, %c0, %c1)            // acc_prev
    ttkernel.sub_binary_tile(%c0, %c1, %c0)     // fwd = acc - acc_prev
    ttkernel.pack_tile(%c0, %6, %c0, true)
    ttkernel.cb_push_back(%6, %c1_i32)

    %8 = arith.addi %arg0, %c1                   // k+1
    ttkernel.experimental::semaphore_wait(%9, %8)  // wait(es, k+1)

    ttkernel.fill_tile(%c0, %cst)               // (recv staging tile)
    ttkernel.pack_tile(%c0, %0, %c0, true)
    // acc_prev = acc
    ttkernel.copy_tile(%5, %c0, %c0); ttkernel.copy_tile(%0, %c0, %c1)
    ttkernel.add_binary_tile(%c0, %c1, %c0)
    ttkernel.pack_tile(%c0, %2, %c0, true)      // -> acc_prev CB
    // acc += r
    ttkernel.copy_tile(%5, %c0, %c0); ttkernel.copy_tile(%1, %c0, %c1)
    ttkernel.add_binary_tile(%c0, %c1, %c0)
    ttkernel.pack_tile(%c0, %5, %c0, true)      // -> acc CB (in place)
  }
  ttkernel.cb_push_back(%5, %c1_i32)            // final acc -> output store
  return
}
```

### Data-movement kernel — the fabric send (NoC0)

Opens one fabric connection around the loop (not per-iteration), and each step
does a `fabric_mcast_fast_write_any_len` of the forwarded tile to the neighbour
plus a `fabric_mcast_sem_inc`. The `if k != 0` guard pops the previous send tile;
the send value is re-produced by compute each iteration so the CB stays balanced.

```mlir
func.func private @datamovement_kernel4() {
  %0 = ttkernel.experimental::create_fabric_connection_manager()
  ttkernel.experimental::setup_fabric_connections(%0)
  ...
  %16 = arith.remsi (%6 + 1), %c4               // nbr = (p+1) % N
  %17 = get_device_id_from_logical_mesh_position(%0, %5, %16)
  scf.for %arg0 = %c0 to %c3 step %c1 {
    scf.if (%arg0 != 0) { noc_async_write_barrier(); cb_pop_front(%4) }
    ttkernel.cb_wait_front(%4, %c1_i32)         // the fwd tile from compute
    %25 = get_noc_addr(%10, %9, %24)
    ttkernel.experimental::fabric_mcast_fast_write_any_len(%0, .., %17, %17, %25, %23, %c4096_i32)
    ttkernel.experimental::fabric_mcast_sem_inc(%0, .., %17, %17, %26, %c1)     // bump nbr's es
    ttkernel.experimental::semaphore_wait(%28, %arg0+1)
  }
  ttkernel.experimental::close_fabric_connections(%0)
  return
}
```

The companion `datamovement_kernel3` (NoC1) does the input read, the
`device_synchronize` semaphore handshake, and the final `out` store.

## Multi-core (grid) variant

`test/d2m-jit/test_all_reduce_grid.py` is the same algorithm distributed over a
Tensix grid — each core `(cy,cx)` all_reduces its sub-block, exchanging with the
*same* core on neighbour devices. It needs `num_links` raised (cores ≤
num_links × 2) and the D2MToTTKernel fix that targets the peer's `my_logical`
core for the scratch fabric write (not a hardcoded `(0,0)`). Hardware ceiling on
this box: 2 eth channels → num_links ≤ 2 → max 4 fabric cores (a 2×2 grid).
```
