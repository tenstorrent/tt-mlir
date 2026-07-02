<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sprint overview — Mesh / fabric CCL in d2m-jit

*Branch `nsmith/d2m-ccl2`, based on `18908b1351` (`origin/nsmith/split-threads`).
~130 commits. Target for a 30-minute review.*

## The one-sentence summary

We brought **multi-device collective communication (CCL)** to the `d2m` dialect
end-to-end: a Python programming model for authoring mesh/fabric kernels
(`all_gather`, ring `all_reduce`, fused all-gather-matmul), the compiler backend
work to lower them through `d2m → ttkernel → ttmetal`, and the on-silicon
bring-up on a 4-chip Blackhole ring. The whole thing is driven from `d2m-jit`,
our Python testbed for the dialect.

**What runs on device today** (1×4 Blackhole ring):
- ring `all_reduce` (single-core, multi-core grid, and a genuine runtime `scf.for` loop)
- `all_gather` + tensor-parallel matmul (AGMM), scaled to a 4×4 gather grid
- `all_reduce` on 30B/70B attention shapes

---

## Agenda (30 min)

1. **The programming model** — mesh/fabric API with examples (~10 min)
2. **Bring-up war stories** — the issues we hit and how we fixed them (~12 min)
3. **Backend pipeline changes** — what moved in the compiler (~8 min)

---

## 1. The mesh / fabric programming model

`d2m-jit` kernels are ordinary Python functions decorated with `@d2m.kernel`.
The body is a *restricted* Python that an AST visitor parses straight into a
`d2m.generic` (it is never executed as Python). Host-scope code builds a lazy
MLIR module; `to_host()` compiles it, runs it on a mesh device, and returns
torch tensors.

The CCL work added three layers on top of the single-device DSL:

### (a) Declare a device mesh + shard host tensors across it

```python
d2m.mesh((1, 4), topology=("linear", "ring"))     # 1×4 device mesh, ring on axis 1

L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1])

# Distribute a full [32, 128] host tensor as four [32,32] shards, one per device.
ins  = d2m.mesh_shard(full, L, shard_dims=[0, 1], shard_shape=[1, 4])
outs = d2m.empty(L)
```

- `mesh_shard` emits `d2m.mesh_shard` (`full_to_shard`); the host sees one full
  tensor, the compiler distributes the shards. `mesh_gather` on `to_host()`
  reverses it (`shard_to_full`).
- The kernel sees one **per-device shard**; the runtime marshals a single full
  borrowed host tensor per argument.

### (b) Cross-device kernel ops: position, sync, fabric send/recv, semaphores

New kernel-body vocabulary (each maps to a `d2m` op):

| Op | What it does |
| --- | --- |
| `mesh_position(dim)` | this device's position on the mesh axis (analogue of `core_index` for cores) |
| `device_synchronize(sem, ...)` | cross-device CCL barrier (receivers signal senders) |
| `remote_store(dst, idx, src, start_device=, device_mcast_shape=, semaphore=)` | **fabric write**: push a shard to a peer device (optionally multicast + increment a peer semaphore) |
| `fabric_recv(buf, [])` | receive a peer-pushed shard into an L1 buffer |
| `semaphore_set/inc/wait` | free-function semaphore handshakes (over `!d2m.global_semaphore`) |
| `core_read` / `core_write` | on-chip core→core L1 read/write (NoC), the intra-device dual of the fabric ops |

Semaphores are a **new kernel-argument kind**: allocate on the host with
`d2m.global_semaphore()`, pass into the kernel, and they arrive typed
`!d2m.global_semaphore` in the body.

### (c) A fabric config on the generic

```python
kernel(ins, outs, start_sem, end_sem, grid=(1, 1),
       fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                routing="unidir_ring_torus"))
```

This attaches a `#ttcore.fabric_connection_config` to the `d2m.generic`, which
tells the backend to open a fabric connection manager and lower the cross-device
ops onto fabric writes / mcast semaphore increments.

### Worked example — runtime-loop ring all_reduce

Each of N devices holds a `[32,32]` shard; after `N-1` fabric steps every device
holds the full sum. This is a genuine runtime `scf.for` with fabric ops inside:

```python
@d2m.kernel
def ring_all_reduce(in0, out, ss, es):
    dy = mesh_position(0)
    p  = mesh_position(1)                 # this device's ring position
    cy, cx = core_index(0), core_index(1)
    device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                        num_receivers=N - 1, core_indices=[cy, cx])
    nbr = (p + 1) % N                      # forward neighbour on the ring

    acc = zeros([1, 1])                    # COMPUTE-owned accumulator init
    acc += remote_load(in0, [0, 0])        # fold in own shard
    acc_prev = zeros([1, 1])
    for k in range(N - 1):                 # runtime scf.for, fabric ops inside
        fwd = acc - acc_prev               # send-only forward (== last received)
        t = empty([1, 1])
        remote_store(t, [], fwd, start_device=[dy, nbr], device_mcast_shape=[1, 1],
                     semaphore=es, semaphore_indices=[cy, 0])
        semaphore_wait(es, k + 1)          # wait for step k's arrival
        r = fabric_recv(t, [])
        acc_prev = copy_(acc_prev, acc)    # snapshot acc
        acc += r                           # accumulate received shard
    remote_store(out, [0, 0], acc)
```

That single generic lowers to **three tt-metal kernels**: a compute kernel (the
loop-carried accumulate), a NoC0 data-movement kernel (opens the fabric
connection once, does `fabric_mcast_fast_write` + `fabric_mcast_sem_inc` per
step), and a NoC1 kernel (input read, barrier, final store).

### Worked example — `is_router_core()`: fabric on a subset of the grid

Fabric can only be issued from a small number of cores per mesh link
(`num_links × cores_per_link`), but a *matmul* wants the whole grid. So a
**fused** all-gather + matmul can't put both roles on every core. `router_cores`
+ `is_router_core()` solve this: declare which cores own the fabric connection,
and let the kernel branch on that role. This is what makes a single generic both
gather over the ring *and* matmul the gathered activations (the AGMM overlap
pattern), instead of two serialized generics.

```python
@d2m.kernel
def fused_agmm(in0, wt, g, out, ss, ready, es):
    cy, cx = core_index(0), core_index(1)
    dy, p  = mesh_position(0), mesh_position(1)
    w = remote_load(wt, [0, cx])              # every core: its TP weight column block

    if is_router_core():                       # only the declared router cores run fabric
        device_synchronize(ss, start_device=[dy, 0], mcast_shape=[1, N],
                            num_receivers=N - 1, core_indices=[cy, cx])
        own = remote_load(in0, [0, 0])
        remote_store(g, [p, 0], own, start_device=[dy, 0],   # mcast own shard to slot p on all devices
                     device_mcast_shape=[1, N], semaphore=es, semaphore_indices=[cy, 0])
        semaphore_wait(es, N - 1)              # all N shards gathered into g[0..N-1]
        semaphore_set(ready, 1, core=[0, 0], mcast=[N, gn])  # release the whole compute grid

    semaphore_wait(ready, 1)                    # every core waits for the gather to finish
    sq = remote_load(g, [cy, 0])                # gathered row cy (NoC read from the router's L1)
    remote_store(out, [cy, cx], sq @ w)         # matmul runs on the full (N, gn) grid
```

```python
fused_agmm(in_s, w_s, g_s, o_s, ss, ready, es, grid=(N, gn),
           fabric=d2m.fabric_config(cluster_axis=1, topology="ring",
                                    routing="unidir_ring_torus",
                                    router_cores=[(0, 0), (1, 0)]))  # forward + backward router
```

- `router_cores` is one `(y, x)` per `(link, direction)` slot (a `unidir_ring_torus`
  ring has `cores_per_link = 2` → a forward and a backward router). It is the
  single source of truth: the compiler gates the fabric connection-manager
  lifecycle to those cores, and the runtime wires fabric to exactly that subset.
- `is_router_core()` is a **compiler-recognized** predicate (not an opaque
  `cx == 0`), so it lowers to a comparison of this core's logical position against
  the configured coords — keeping the kernel branch, the lowering, and the runtime
  in agreement. A router core can also call `router_direction()` to learn which
  slot it serves and address the correct forward/backward neighbour.
- Compute cores never hold the fabric connection, so their NoC read-back of the
  gathered data safely sees the fabric write (the read-back vs. fabric-connection
  hazard that broke the single-core fused kernel).

Runs on the full 1×4 ring (`test/d2m-jit/test_all_gather_matmul_fused.py`),
scaling to K=1024 / Nout=4096 with the TP weight distributed across the grid.
Design rationale: [fabric_router_cores_design.md](fabric_router_cores_design.md).

Full spec: [LANGUAGE_SPEC.md](LANGUAGE_SPEC.md). CCL design:
[CCL_SPEC.md](CCL_SPEC.md), [all_reduce_design.md](all_reduce_design.md). Kernel
walkthrough with the lowered IR: [ring_all_reduce_walkthrough.md](ring_all_reduce_walkthrough.md).

---

## 2. Bring-up war stories (the issues we hit)

Multi-device fabric was almost entirely untested through this front, so most of
the sprint was root-causing subtle deadlocks and miscompiles on silicon. The
recurring theme: **the IR looks correct and static analysis says "identical to
the working path," but it hangs on device.** These are the ones worth showing.

### The all_gather hang — the fabric was never enabled at runtime

The headline story. A hand-built 1×2 `all_gather` compiled, lowered, and
matched the rewriter's kernel byte-for-byte, yet **deadlocked on device** — the
end semaphore stuck at 0. We chased this through: watcher waypoints (localized to
the end `semaphore_wait`), a wait-value sweep, EmitC slot tracing, a flatbuffer
program diff (which found the DSL program was *missing a compute kernel*, a red
herring), and NOC-coordinate analysis.

**Actual root cause:** `_execute` opened the mesh device **without calling
`set_fabric_config`**, so the device fabric defaulted to `DISABLED`. Every
cross-device fabric op (mcast sem-inc, fabric write, semaphore increments)
silently no-op'd, so the semaphore never advanced. Fix: track when a `fabric=`
kernel is used and call `set_fabric_config(FABRIC_1D_RING)` before
`open_mesh_device`. Lesson recorded: it wasn't the kernel, the program, or the
compute-kernel split — it was one missing runtime call.

### Line all_gather deadlock — fabric routing off-by-one on a 2-device line

Getting the ring `all_gather` to work as a *linear* topology on a 2-chip n300
surfaced four real fabric bugs, all fixed:
- `D2MAllGatherRewriter` was ring-only; taught it to accept linear topology and
  pick routing mode by topology.
- **Line-mcast routing:** the "remove my_idx from endpoints" remap *inverted* the
  send direction for the end-of-line sender on a 2-device range (it sent forward
  into a non-existent link, so the increment never arrived). Confined the remap
  to the wrapped/torus branch.
- Edge devices legitimately have one direction = `COUNT`; gated the direction
  asserts so they don't trip under watcher.
- **End-semaphore off-by-one:** `remote_store` increments the end semaphore on
  every device in the mcast range *including the sender* (a local self-inc), so
  each device gets `num_devices` increments, but the wait targeted
  `num_devices - 1`. Fixed the wait target.

### Multi-step ring deadlock — loop fusion hoisted the semaphore_wait

A single ring step ran, but 2+ steps hung. Not a semaphore-count bug — the
trigger was `InsertSpillAndScratch`'s loop fusion **hoisting a `semaphore_wait`
above the compute nest that produces the value it waits on**. Added a barrier so
the wait can't move above its producer. Recv was also pivoted to a gridless
`#l1` tmp scratch via a dedicated `fabric_recv` op (view-free) to avoid a NoC
read-back on the same NoC the fabric connection holds.

### Loop-carried accumulator miscompute — FIFO page split

A `for k: acc += ...` accumulator inside a user loop silently miscomputed. The
carry is an `scf.for` iter_arg that must stay on one CB page. Two rules fell out
(now enforced by the DSL):
- The accumulator init must be **compute-owned** (`acc = zeros(...)`); a
  DM-seeded init (`acc = remote_load(...)`) puts init and compute-pack on
  different FIFO pages and garbles the carry. The DSL now *rejects* the DM-seed
  pattern at trace time.
- `acc += x` lowers via an in-place `__add_acc__` (the eltwise dual of matmul's
  `c += a@b`) so the carry bufferizes as an iter_arg instead of failing
  one-shot-bufferize; `copy_` is an explicit in-place copy that survives
  canonicalization.

### Chunked matmul garbaged chunks t>0 — wrong L1-accumulate guard

A fused matmul streamed in chunks inside a user `scf.for` all-gathered chunk 0
correctly but garbaged every later chunk. Root cause in
`InsertDstRegisterAccess`: the matmul's L1-accumulate guard selected the
**enclosing per-chunk streaming loop** as the reduction loop, so it accumulated
onto stale content each iteration. Fix: exclude loops whose IV drives the
generic's output-store index (parallel/streaming loops) from L1-acc guard
selection, keying the accumulate on the matmul's own K loop.

### Cross-device read-back rejected by bufferization

Reading back an operand a peer pushed into (`remote_load(out, …)` after a
cross-device `remote_store(out, …)`) failed one-shot-bufferize
(`memref/tensor must be remote`). A cross-device store was modeled as a *local*
memory write, creating a false read+write conflict that dropped the device
layout. Fix: `RemoteStoreOp::bufferizesToMemoryWrite` returns false for the
memref operand when `startDevice` is non-empty (a cross-device store writes a
*remote* buffer; the local recv is ordered by semaphores, not local SSA).

### TensorAccessor DMA dropped fabric mcast / mis-addressed DRAM

Two accessor-DMA correctness bugs: the accessor write path dropped the
cross-device fabric mcast (block-diagonal all_gather), and on DRAM operands the
accessor mis-addressed the paged buffer (PCC 0.0 — magnitudes right, tiles
permuted). Fixes: restore the fabric mcast in the accessor path; and route DRAM
remote operands to the proven fully-indexed DMA form, keeping the validated
L1-only accessor path for L1 shards.

### The "send-only" CB rule (design constraint, not a bug fix)

A value that is fabric-sent must be **send-only** — a CB consumed by both a DM
`remote_store` and a compute op fails to legalize (unresolved memref→cb cast),
and a value feeding two DM ops deadlocks (first consumer pops the CB, second
hangs on `cb_wait_front`). So we forward received values as fresh send-only
compute outputs (e.g. send `acc_k − acc_{k-1}` == the received shard). This
shaped every ring/all-gather kernel.

> There is also a **pre-existing environment issue** (not from this branch):
> multi-device fabric tests time out at device-open on this specific box. It
> masks `test_mesh`/`test_semaphore`; the CCL kernels themselves are validated
> via the tests that do run on the ring.

---

## 3. Backend pipeline changes

The work touched the whole `d2m → ttkernel → ttmetal` lowering. Highlights
(`~3,100` lines across `lib/`/`include/`):

### New dialect ops (`D2MGenericRegionOps.td`, +287 lines)
- `mesh_position`, `device_synchronize` — mesh position + cross-device barrier.
- `fabric_recv` — dedicated view-free fabric receive into `#l1` scratch.
- `core_read` / `core_write` — on-chip core→core L1 primitives (NoC), plus a
  proposed MPI-style core-collective surface built on them.
- `is_router_core` / `router_direction` — fabric router-core queries (run fabric
  on a *subset* of a generic's grid).
- Cross-device kwargs surfaced on `remote_store` / `remote_load` (start_device,
  device_mcast_shape, semaphore, semaphore_indices).

### Thread splitting — `SplitUnifiedThreadV2` (new, ~820 lines)
A dataflow-model rewrite of the unified-thread splitter (A/B toggle via
`config.use_split_unified_thread_v2`). The v1 splitter asserted on kernels with
multiple synchronizable ops (`device_synchronize` + `remote_store` +
`semaphore_wait` in one region) — exactly the CCL kernel shape. v2 handles it,
recognizes `fabric_recv` as an input-CB producer, and keeps the fabric chain on
a single datamovement thread.

### Fabric lowering — `D2MToTTKernel` (+600 lines)
- **Fabric connection manager**: created for funcs with a fabric write,
  fabric semaphore op, **or** a `mesh_position` (a mesh_position-only thread
  still needs the fcm).
- Cross-device `remote_store` → `fabric_mcast_fast_write` + `fabric_mcast_sem_inc`;
  local-CB destinations use `GetWritePtr` (not the remote cast path).
- `device_synchronize` / `mesh_position` / cross-device semaphore lowering.

### Scheduling / synchronization passes
- `ScheduleDMA` — pin the CCL start barrier (and explicit semaphore inc/set) to a
  single datamovement thread.
- `InsertSpillAndScratch` — barrier so loop fusion can't hoist a `semaphore_wait`
  above its producing compute nest.
- `MarkSynchronizedBuffers` — single-buffer loop-carried accumulators.
- `InsertDstRegisterAccess` — exclude output-store-indexing (streaming) loops
  from matmul L1-accumulate guard selection.
- `Allocate` — handle loop-carried memref accumulators; null-safe checks for
  in-loop / loop-carried `remote_store` (block-arg local buffers).

### DMA lowering
- **TensorAccessor-based DMA path** (new), enabled by default, coexisting with
  mcast/local DMAs; page-granularity fix for row-major buffers; guarded against
  shard-level ops.
- `LowerLoadStoreOpsToDMA` — lower **implicit-form** `remote_load`/`remote_store`
  (no CB) directly to `dma_read`/`dma_write` (fixes a datamovement-CCL segfault).
- `LowerDMAToFullyIndexedForm` — route DRAM remote operands to fully-indexed DMA.

### Fabric routing / runtime (tt-metal LLKs + flatbuffer)
- `experimental_fabric_topology_info.h` / `experimental_fabric_1d_routing.h` —
  the line-mcast direction-remap and conditional-direction-assert fixes above.
- `TTMetalToFlatbuffer` — `BufferDistributionSpec` for all sharded L1 buffers;
  `router_cores` wired through the flatbuffer so runtime fabric setup targets
  just the router subset; fabric-config serialization.
- `d2m-jit` runtime: `set_fabric_config` around mesh open/close (the all_gather
  fix); mesh-shape plumbing to `ttcore-register-device`; `#ttcore.tensor_mesh`
  boundary tagging so the runtime sizes distributed host buffers to match the
  mesh device buffer.

---

## Where to dig deeper

| Topic | Doc |
| --- | --- |
| Programming model (normative) | [LANGUAGE_SPEC.md](LANGUAGE_SPEC.md), [README.md](README.md) |
| CCL all_gather design + log | [CCL_SPEC.md](CCL_SPEC.md) |
| Ring all_reduce design + root causes | [all_reduce_design.md](all_reduce_design.md), [ring_all_reduce_walkthrough.md](ring_all_reduce_walkthrough.md) |
| Fused all-gather + matmul (AGMM) | [agmm_design.md](agmm_design.md), [fused_matmul_allgather_8x8_design.md](fused_matmul_allgather_8x8_design.md) |
| core_read / core_write | [core_read_write_spec.md](core_read_write_spec.md) |
| Fabric router cores | [fabric_router_cores_design.md](fabric_router_cores_design.md) |
| Rebase regression triage (R1–R18) | [../../REBASE_REGRESSIONS.md](../../REBASE_REGRESSIONS.md) |
