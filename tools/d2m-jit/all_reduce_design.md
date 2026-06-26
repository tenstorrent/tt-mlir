# d2m-jit `all_reduce` — design

Design for a fabric `all_reduce(sum)` in d2m-jit, building on the validated
all_gather path (`test_all_gather_1x4_roundtrip`,
`test_streaming_matmul_all_gather_1x4`) and the recently-fixed accumulate-in-a-
loop codegen. Companion to `CCL_SPEC.md` (all_gather), the fused-matmul design,
and `STATUS_fused_matmul_allgather.md`.

_Target chosen: **ring** (bandwidth-optimal). gather+reduce is kept only as a
correctness reference (§5)._

## Semantics

Every device `d` holds a same-shaped tensor `X_d` (M×K tiles). After
`all_reduce(sum)`, every device holds `Y = Σ_d X_d` (same shape, replicated).

Contrast with all_gather: all_gather inputs are *distinct shards* and the output
is their *concatenation*; all_reduce inputs are *full, same-shape* tensors and
the output is their *elementwise sum*. So all_reduce = (a collective that moves
data on the fabric) + (a reduction). The fabric machinery is the same; the new
ingredient is the per-element sum.

## Host framing (reuses all_gather — no new mesh primitive)

`mesh_shard` is shard-only (full→shard by division along `shard_dims`); there is
no explicit "replicate". But the all_gather framing already hands each device a
full-shaped operand, so we reuse it verbatim:

- **Inputs:** build `full_X` of shape `(M, N*K)` where column-block `d` is the
  per-device tensor `X_d`. Then
  `mesh_shard(full_X, L_in, shard_dims=[0,1], shard_shape=[1,N])` gives device
  `d` exactly `X_d` (M×K) — identical to how the all_gather tests feed `A_d`.
- **Output:** every device computes the same `Y` (M×K). Lay it out like
  all_gather's replicated output and `mesh_gather(..., shard_shape=[1,N])`
  column-concats the N identical device copies → `(M, N*K)`; the test checks
  every K-block equals `Σ_d X_d`.

So the host side is a drop-in of the all_gather test scaffold. The only new code
is the kernel(s).

## Ring algorithm (the target)

The 4-chip Blackhole forms a ring; `fabric_config(topology="ring")` is available.
Standard ring all_reduce = **ring reduce_scatter** then **ring all_gather**, each
N−1 steps, optimal `2(N−1)/N` payload.

Split each `X_d` into N chunks along a chunk axis: `X_d = [c_d^0 … c_d^{N-1}]`
(for the first cut, chunk = 1 tile, so `X_d` is N tiles).

**Phase R — reduce_scatter (N−1 steps).** With device index `p = mesh_position`,
each device sends one chunk to its `+1` neighbor and accumulates the chunk it
receives from its `−1` neighbor:

```
acc[i] = c_p^i                                   # local copy of every chunk
send_idx = p
for step in range(N-1):
    recv_idx = (p - 1 - step) mod N              # chunk index arriving this step
    # send acc[send_idx] to neighbor (p+1); receive partial for recv_idx
    remote_store(neighbor=p+1, chunk=acc[send_idx])
    partial = <received chunk recv_idx from p-1>
    acc[recv_idx] = acc[recv_idx] + partial      # accumulate-on-receive (tile add)
    send_idx = recv_idx
# after N-1 steps device p owns the fully-reduced chunk r = acc[(p+1) mod N]
```
After phase R, device `p` owns one fully-reduced chunk `r_p = Σ_e c_e^{(p+1)%N}`.

**Phase G — all_gather (N−1 steps).** Circulate the reduced chunks around the
ring so every device ends with all N reduced chunks = full `Y`. This is exactly
the existing ring all_gather, applied to the reduced chunks — reuse the validated
streaming all_gather kernel.

So the *new* work is **phase R**; phase G is our existing all_gather.

### Key new fabric usage: unicast-to-neighbor

all_gather uses `remote_store(start_device=[dy,0], device_mcast_shape=[1,N])` —
an mcast to the whole line. A ring step is a **unicast to the +1 neighbor**:
`start_device=[dy, (p+1)%N]`, `device_mcast_shape=[1,1]`. Confirming this
single-neighbor destination addressing (and the matching `device_synchronize` /
end-semaphore counts for a 1-receiver send) is the first thing to validate — it
is the one primitive the all_gather path never exercised.

### Single-fabric-core + accumulate-on-receive

The ring uses one link direction → one link core (the single-fabric-core rule is
satisfied as in all_gather). Accumulate-on-receive is a `tile add` of the
arrived chunk + the local accumulator, inside the per-step `scf.for` — i.e. the
streaming-loop + in-loop-accumulate pattern unblocked by the recent
`InsertDstRegisterAccess` fix (`genericOutputStoreDependsOnIV`). Worth running
the loop-accumulate early to confirm the fix generalizes from matmul to eltwise.

## Incremental milestones

1. ✅ **Unicast-to-neighbor probe — DONE (2026-06-24).** `_ring_probe.py`:
   device `p` sends one tile to `(p+1)%4`, receives from `(p-1)%4`. Verified on
   the 1×4 Blackhole ring — every device ends with its `(p-1)%4` neighbor's tile,
   **including the wraparound 3→0** (maxdiff ~0.0019, the usual all_gather
   tilize/store artifact). Confirms: `start_device=[dy,(p+1)%4]` +
   `device_mcast_shape=[1,1]` expresses a single-neighbor send; in-kernel
   `(p+1)%4` lowers fine (`arith.remsi`); ring wraparound routes correctly with
   `topology="ring", routing="unidir_ring_torus"`; `num_receivers` /
   `semaphore_wait(end_sem, 1)` are right for a 1-receiver, no-self-inc send.
   The one unproven primitive is now proven — the rest of the ring is bookkeeping.
2a. ⛔→✅ **In-kernel read-back of a pushed operand — was BLOCKED, now UNBLOCKED
   (2026-06-24).** The ring's accumulate-on-receive must **read, in-kernel, the
   tile a neighbor pushed** into a shared operand. Probes (`_rs_probe.py`) first
   confirmed the same wall as the fused-matmul "probe 3", cross-device flavor:
   recv-as-second-output → `'d2m.generic' op must have exactly one output
   operand`; recv-as-slot-of-the-one-output read via `remote_load(out,…)` then
   `remote_store(out,…)` → `'d2m.remote_store' op memref/tensor must be remote
   (have a device layout)`.

   **Root cause:** `ttcore-one-shot-bufferize`. A cross-device `remote_store`
   was modeled as a *local* memory write of its memref operand, so reading that
   operand back in-kernel created a false read+write conflict; bufferize copied
   the store out-of-place into a fresh `#l1` buffer that dropped `#ttcore.shard`.

   **Fix (chosen path (c), `RemoteStoreOp::bufferizesToMemoryWrite` in
   `D2MGenericRegionOps.cpp`):** a store with `startDevice` set writes a *remote*
   device's buffer, not the local operand (the local recv side is written
   externally by peers, ordered by semaphores), so it no longer registers a
   local write. The false conflict disappears; the read-back stays in-place with
   the device layout. Validated: `_rs_probe.py` passes on the 1×4 ring (single
   output, slot 0 = recv read-back, slot 1 = result); the full 1×4 fabric +
   matmul + semaphore suite is regression-free (only the pre-existing
   `test_mcast_overwrite_grid_2x2` fails). **This also unblocks the
   single-generic fused matmul+all_gather** (same probe-3 wall).

   Use a **single output with slots** (slot 0 = ring recv, slot 1 = result) —
   the "exactly one output operand" rule still holds.

2b. ⚠️ **reduce_scatter — compiles & 1-step runs; multi-step HANGS (2026-06-24).**
   `_rs_full.py`: running-partial ring, straight-line unrolled (N-1=3 steps; a
   Python `for` in a kernel becomes `scf.for`, whose loop-carried *tensor*
   `partial` hits `Yield operand not equivalent to iter bbArg` in bufferize — so
   unroll instead), single output `out` grid [N,1] (slots 0..N-2 = per-step recv,
   slot N-1 = result), `c_p[idx]` dynamic input reads + recv read-back via the
   now-unblocked output slot. **It compiles** (the read+write unblock holds
   end-to-end) and a **1-step** version runs correctly-structured on the 1×4 ring.
   But **2+ steps hang** — and this is NOT the semaphore-count strategy:
   cumulative `wait(es, t+1)` hangs, per-step `wait(es,1,reset=0)` segfaults,
   and a separate end-sem per step also hangs.

   **Localized blocker:** multiple cross-device fabric sends *interleaved with*
   `semaphore_wait` across steps deadlock. The ring is inherently sequential
   (step t+1's send value depends on step t's recv+accumulate), so interleaved
   send→wait→send is unavoidable. This matches the streaming-all_gather
   constraint ("multiple fabric sends must stay on one DM thread / one
   `fabric_connection_manager` or the two managers deadlock"): the interleaved
   `semaphore_wait` appears to make `ScheduleDMA` split the fabric sends across
   DM threads. The single-step case (one send, then wait) avoids it.

   **Root cause localized (2026-06-24, FINAL): the 2nd in-kernel READ-BACK of the
   output operand, NOT the send count.** Minimal repro:
   `test/d2m-jit/repro_ring_fabric_readback_hang.py`. Clean elimination on the
   1×4 ring (single fabric DM thread):

   | kernel | fabric sends | output read-backs | result |
   |---|---|---|---|
   | forward inputs, no read-back | 2–3 | 0 | completes |
   | compute add of inputs, no read-back | 2 | 0 | completes |
   | ring, 2 steps | 2 | **1** | completes |
   | ring, 3 steps | 3 | **2** | **hangs** |

   So it is NOT the send count (3 sends, no read-back, complete — matching
   streaming/large-block all_gather doing up to 16 sends fine), NOT the compute,
   NOT the interleaved wait. The trigger is the **second `remote_load` read-back
   of the output operand**, which lowers to a `noc_async_read` on **NoC0** — the
   same NoC the fabric connection holds.

   ttkernel-IR confirmation (`_ttk_dump.py`, 2-send vs 3-send): the fabric
   connection is opened once / closed once (no per-send churn), and each step's
   codegen is identical (`fabric_mcast_fast_write_any_len` → `fabric_mcast_sem_inc`
   → conditional self-inc → `semaphore_wait`). So this is a **runtime resource
   conflict**, not a codegen bug — one output read-back interleaved with the open
   fabric connection on NoC0 is fine; the second deadlocks (suspected NoC0
   command-buffer / transaction-id / CB-depth contention between the local
   read-back and the fabric connection on the same NoC). Pinning the read-back to
   NoC1 (the other DM thread) is blocked because the read-back feeds the compute
   that feeds the next fabric send — a producer-consumer chain ScheduleDMA can't
   split across DM threads (no DM↔DM CB handshake).

   **Fix direction (re-validated):** eliminate the DM read-back of the output —
   model the cross-device fabric **recv as a compute-consumed input CB**
   (`cb_wait_front` on incoming; the fabric write + its semaphore are the
   producer), exactly the tt-metal CCL receiver pattern. This removes the NoC0
   read-back entirely. A deeper lowering change, but now well-motivated and
   specific. (Alternative: two-generic-per-step, which also avoids interleaving
   read-backs with the open fabric connection on one thread.)
3. **Combine with all_gather → all_reduce.** Phase R + existing ring all_gather;
   verify every device holds `Y = Σ_d X_d`. Regression guard
   `test_all_reduce_1x4_roundtrip` (gated ≥4 devices).
4. **Scale chunks > 1 tile / larger shards** (block-packed, mirroring the
   all_gather large-block work), then DRAM-staged output if it exceeds L1.

## Reference fallback — gather+reduce (NOT the target)

For debugging only: all_gather every `X_d` to all devices (each device ends with
`[X_0…X_{N-1}]` stacked), then a fabric-free compute generic sums the N copies.
Correct but ~N× the optimal bandwidth. Use it to bisect if the ring misbehaves
(it isolates "is the reduction/host-framing right" from "is the ring right"),
exactly as the zeros/eltwise siblings did for the fused matmul.

## Risks / open questions

- **Unicast-to-neighbor addressing** (milestone 1) — the one unproven primitive;
  everything downstream depends on it.
- **Chunk-index bookkeeping** in phase R is the classic ring footgun; the
  milestone-2 PCC check on the per-device owned chunk pins it down before
  combining with phase G.
- **Per-step semaphores / double-buffering** — each ring step needs its own
  start/end sync so step t+1 doesn't clobber step t's buffer; mirror the
  all_gather end-semaphore counting (1 remote + 1 self per send) per step.
- **N=2 sub-mesh can't train fabric on this box** — validate on the full 1×4
  ring only (see `STATUS_fused_matmul_allgather.md`).
- **f32 exactness:** the reduction is plain `tile add` (exact in f32), so unlike
  the matmul cases use a tight abs-diff in addition to PCC.

---

## Milestone 3 scope: fabric-recv as a compute-consumed input CB

The blocker root-caused in milestone 2b is the in-kernel DM read-back of the
output operand (`remote_load(out,…)` → `noc_async_read` on NoC0) interleaved with
the open fabric connection on the same NoC; the **2nd** such read-back deadlocks.
The fix is to **never read the recv back on the DM thread** — instead land the
incoming fabric write in a circular buffer that the **compute thread consumes via
`cb_wait_front`**, exactly the tt-metal CCL receiver pattern. The fabric write +
its semaphore increment become the CB's *producer*; the accumulate reads it as a
normal compute input. This removes the NoC0 read entirely, so the NoC0 contention
that deadlocks at the 2nd read-back cannot occur.

### Target dataflow (per ring step, receiver side)

Today (hangs):
```
DM:      remote_store(out[t], partial) --fabric--> peer   ;  semaphore_wait(es_t,1)
DM:      recv = remote_load(out[t])    <-- NoC0 read-back of the output operand  ✗
compute: partial' = recv + c
DM:      remote_store(out[t+1], partial') --fabric--> peer
```
Target (no DM read-back):
```
peer --fabric write--> my recv CB[t]   ;  peer --fabric sem_inc--> my recv CB sem
compute: cb_wait_front(recvCB)         <-- consume the arrived tile, no NoC read   ✓
compute: partial' = front(recvCB) + c  ;  cb_pop_front(recvCB)
DM:      remote_store(out[t+1], partial') --fabric--> peer
```
The DM thread now only *sends*; the *receive* is a compute-side CB wait. The
sender already lowers to `fabric_mcast_fast_write_any_len` + `fabric_mcast_sem_inc`
(D2MToTTKernel); the change is what the **destination** of that write/sem-inc is
(a recv CB + its front semaphore) and how the **receiver** consumes it.

### Pieces to build

1. **A "fabric recv CB" operand kind / marker.** A generic operand that is (a)
   the destination of peers' cross-device `remote_store`s and (b) a
   compute-consumed input CB (`cb_wait_front`/`cb_pop_front`), whose producer
   pointer is advanced by the incoming fabric semaphore rather than a local
   `cb_push_back`. This is the crux: it couples "cross-device write target"
   (output-like) with "compute input CB" (input-like) — an in-out/aliased
   operand. Builds on the bufferization groundwork already landed
   (`RemoteStoreOp::bufferizesToMemoryWrite` treats a cross-device store as a
   non-local write; see [[d2m-crossdevice-store-readback]]).

2. **DSL surface (`api.py` / `_src/ast.py`).** A kernel-body primitive for the
   receiver, e.g. `recv = fabric_recv(recv_cb, [t])` that the AST lowers to a
   compute-side CB consume, replacing the `remote_load(out,[t])` read-back. The
   sender keeps `remote_store(..., start_device=, semaphore=recv_cb_sem)` but its
   semaphore now targets the recv CB's front semaphore.

3. **D2MToTTKernel lowering.** Receiver consume → `cb_wait_front(recvCB, 1)` then
   the tile feeds the existing compute (tile_add) → `cb_pop_front`. No
   `remote_load`/`noc_async_read` is emitted for the recv. Sender's
   `fabric_mcast_sem_inc` targets the recv CB's pages/front semaphore so the
   peer's `cb_wait_front` observes it. The ttkernel ops needed already exist
   (`cb_wait_front`, `cb_pop_front`, `fabric_mcast_fast_write_any_len`,
   `fabric_mcast_sem_inc`); the work is binding the CB-front wait to the
   fabric-incremented semaphore (verify no new ttkernel op is required — if the
   CB front pointer can't be driven by a remote sem_inc directly, a small
   `cb_wait_front`-on-external-semaphore shim op may be needed).

4. **Split / ScheduleDMA interaction.** With the recv consumed by compute, there
   is no DM read-back op, so `split-unified-thread-v2` sees the recv CB as a
   compute input (CBComputeInfo.consumed) with a *fabric* producer (no DM load
   partner). ScheduleDMA still pins the (send-only) fabric ops to one DM thread;
   the NoC0 read-back is simply gone. Confirm the split's CB handshake handles a
   compute-input CB whose producer is the remote fabric write (not a local
   `remote_load`).

### Reuse vs new

- **Reuse:** sender path (`fabric_mcast_fast_write_any_len` + `fabric_mcast_sem_inc`),
  the cross-device-store-non-local-write bufferization fix, compute tile_add,
  `cb_wait_front`/`cb_pop_front`, the unicast-to-neighbor addressing (milestone 1).
- **New:** the recv-CB operand kind/marker (#1), the `fabric_recv` DSL primitive
  (#2), the receiver-consume lowering + sem→CB-front binding (#3).

### Milestones

3a. **Receiver-CB primitive, single step (no ring).** Two devices (or 1×4 with a
    1-hop send): peer fabric-writes my recv CB[0] + sem-incs; my compute
    `cb_wait_front`s it and stores it out. Verify correctness AND that no
    `noc_async_read` of the recv is emitted (check ttkernel IR). This is the
    milestone-2a read-back probe re-expressed via a compute-CB consume.
3b. **Two ring steps via recv-CB.** The case that hangs today (2 read-backs)
    should now pass, since there is no NoC0 read-back. This is the decisive test.
3c. **Full reduce_scatter via recv-CB** (N-1 steps), then combine with phase G
    (existing all_gather) → `test_all_reduce_1x4_roundtrip`.

### Risks / open questions

- **Can a CB's front/pages pointer be advanced by a remote fabric `sem_inc`?**
  tt-metal CCL receivers do exactly this, but confirm the d2m ttkernel lowering
  can express "cb_wait_front whose readiness is a fabric-incremented semaphore"
  without a new op; if not, scope a minimal shim op.
- **In-out/aliased operand:** the recv CB is written cross-device and read by
  compute. The one-output-operand rule and the operand model must allow a
  compute-input CB that is also a fabric-write destination. May need the recv CB
  to be a dedicated operand kind distinct from the single output.
- **CB depth vs ring steps:** one recv CB with depth ≥ pipeline depth, or a CB
  slot per step. Depth interacts with how far ahead peers may send.
- **Ordering:** `cb_wait_front` replaces `semaphore_wait`; ensure the start
  barrier (`device_synchronize`) and the per-step ordering still compose.

### Milestone 3 progress (2026-06-24)

- **No shortcut confirmed.** There is no pure-L1 (no-NoC) read path
  (`core_read` also lowers to `createNocAsyncRead`), and `ttkernel.noc_async_read`
  has no per-op NoC-index override (`getKernelNocIndex` = `1 - processorIndex`,
  uniform per DM thread). The fabric thread is processor 1 → NoC0, so *any*
  DM-thread read-back of the recv uses NoC0 and contends with the fabric. The
  fix must therefore eliminate the read-back, not relocate it.
- **Foundation validated: recv as a generic INPUT operand.**
  `_recv_input_probe.py` — a cross-device `remote_store` targeting a generic
  *input* operand (the peer's recv), with the receiver reading its own recv,
  compiles and runs correctly on the 1×4 ring (1 step). So the recv can be a
  separate input operand (a compute-input CB) distinct from the result output
  operand — the "one output operand" rule is satisfied, and the cross-device
  store to an input bufferizes fine (builds on the cross-device-store-non-local-
  write fix). NOTE this alone does NOT fix the hang: a `remote_load(recv)`
  read-back is still a NoC0 read; recv-as-input is the clean *foundation* for the
  no-NoC consume below.

**Refined design (recv as input CB, no-NoC consume):**
  - `recv` is a generic **input** operand = a compute-input CB at a uniform
    (fabric-addressable) L1 offset. The peer's `remote_store(recv, start_device=…)`
    fabric-writes directly into that CB's slot, and its `fabric_mcast_sem_inc`
    signals arrival.
  - Receiver consume lowers to: `semaphore_wait` (arrival) → `cb_push_back(recv)`
    — a pointer update, **no `noc_async_read`** — then compute `cb_wait_front` /
    `cb_pop_front`. This removes the NoC0 op that deadlocks at the 2nd read-back.
  - Remaining work (the hard part): (1) a DSL primitive / lowering so consuming a
    fabric-written input operand emits `sem_wait + cb_push_back` instead of
    `remote_load`→`noc_async_read`; (2) coordinate the fabric write target address
    with the recv CB's reserve/push pointers (the fabric must write into the slot
    the CB front will expose — deterministic uniform offset, depth ≥ pipeline);
    (3) confirm split-v2 treats the recv CB as a compute input whose producer is
    the fabric (no DM load partner) and ScheduleDMA leaves the (send-only) fabric
    ops pinned while the recv consume needs no NoC.

### Step 1 implementation finding (2026-06-24): needs a dedicated op

Attempting to implement the no-NoC consume by marking a `remote_load` (attr
`d2m.local_recv`) and skipping its `dma_read` does NOT work: `remote_load` couples
its result CB to a *separate* localBuffer operand (the `dma_read` copies
`src -> localBuffer`'s CB, and compute reads that copy). Skipping the `dma_read`
leaves compute reading an unfilled localBuffer CB, not the fabric-written recv.
Passing the recv operand itself as the localBuffer gives the wrong result type
(the full sharded operand vs the per-tile shard) and conflates two operand CBs.

So step 1 requires a **dedicated `d2m.fabric_recv` op** whose *result is the recv
operand's own CB front* (no separate buffer, no copy):

```
%tile = d2m.fabric_recv %recv[%t]      # recv: a generic INPUT operand (device layout)
```
- **Semantics:** the tile at `recv[t]` has already been written into recv's CB by
  a peer's cross-device `remote_store` (arrival gated by the existing
  `semaphore_wait`); `fabric_recv` exposes it to compute. Compute consumes the
  result via the normal input-CB `cb_wait_front`/`cb_pop_front`.
- **Lowering (LowerLoadStoreOpsToDMA / D2MToTTKernel):** `reserve(recv_cb)` →
  `push(recv_cb)` — NO `dma_read`, NO `noc_async_read`. This is the whole point:
  no NoC op on the fabric thread.
- **split-v2 (`insertComputeCBOpsV2`):** treat `fabric_recv`'s recv CB as a
  compute-consumed input whose *producer is the DM-side push* (not a `dma_read`).
  It is the input-CB `dmLoad`-equivalent for the wait/pop handshake.
- **Bufferization:** `fabric_recv` reads nothing locally (the recv operand is
  written by peers, like the cross-device store); result aliases the recv CB.
- **DSL (`api.py`):** a `fabric_recv(recv, [t])` `@syntax` primitive emitting the
  op; the kernel does `s = fabric_recv(recv, [t]) + remote_load(in0, [idx])`.

Files to touch: `D2MGenericRegionOps.td` (op def + verifier + BufferizableOp
interface), `D2MGenericRegionOps.cpp` (interface impl), `SplitUnifiedThreadV2.cpp`
(recognize it as an input-CB producer), `LowerLoadStoreOpsToDMA.cpp` (reserve→push),
`api.py` (primitive). Validate on milestone 3a (1 step: confirm the ttkernel IR
emits NO `noc_async_read` for the recv, and the result is correct), then 3b (the
2-read-back case that hangs today should pass), then 3c (full reduce_scatter).

This is a focused multi-file op-implementation effort; the foundation (recv as an
input operand, cross-device store into it) and the precise lowering shape are now
pinned, so it can be built directly against this plan.

### Step 1 build status (2026-06-24)

Parts 1-3 are committed and compile (additive; existing all_gather / fused-matmul
/ matmul suites still pass):
- **Part 1** (`da8adf193`): `d2m.fabric_recv` op (verifier, BufferizableOpInterface,
  ShardDMA/Synchronizable interfaces).
- **Part 2** (`5dd2e2cba`): split-v2 recognition -- convert to explicit-CB form on
  the memref operand's own CB; `CBComputeInfo.dmRecv` makes it an input-CB producer
  (compute gets wait/pop).
- **Part 3** (`9ce8a90ca`): `LowerLoadStoreOpsToDMA` reserve+push (no `dma_read`);
  `api.py` `fabric_recv` primitive; bufferize exposes the shard via
  `memref.reinterpret_cast`; split-v2 view-walk follows `reinterpret_cast`.

**Milestone 3a is NOT green yet — one blocker.** The bufferized
`memref.reinterpret_cast` of the `#ttcore.shard` recv operand survives to
`D2MToTTKernel`, which fails to legalize it (source still the `#shard` operand --
the view wasn't rewired to recv's CB nor erased by split). Root issue: `#shard`
operands aren't memref-viewable (subview's stride inference asserts;
reinterpret_cast lowers nowhere), and `remote_load` only sidesteps this by
copying. Repro: `test/d2m-jit/_m3a.py` (1-step recv via `fabric_recv`).

**Next options to unblock 3a:**
1. In split-v2, after the input-CB rewiring, ensure the recv view (reinterpret_cast)
   is rewired to / replaced by the CB wait result and erased on BOTH threads (the
   dead DM-side clone currently survives) -- then the compute reads the CB directly
   and no reinterpret_cast reaches D2MToTTKernel.
2. Or add a `D2MToTTKernel` rewriter for `memref.reinterpret_cast` of a CB
   (resolve to the CB read pointer, like `MemRefSubviewRewriter`).
3. Or a shard-exposure mechanism that doesn't emit a memref view at all (let
   split-v2 map the fabric_recv result directly to `getOrCreateCB(recv)` without a
   bufferize-time view).
Option 1 or 3 is cleanest (no new conversion surface); the grid-[N,1] dynamic
recv-slot offset (currently hardcoded 0) is a separate follow-up after 3a.

### 3a blocker re-diagnosed (2026-06-24): option 1 is moot; need a no-view bufferize

The bufferized `memref.reinterpret_cast` of the `#shard` recv operand fails
**pre-split** -- a be-pipeline dialect-conversion pass before
`d2m-split-unified-thread-v2` rejects `memref.reinterpret_cast` ("failed to
legalize"). Confirmed by running be-pipeline up to (not including) split: it
already fails. So option 1 (erase the view in split-v2) cannot help -- split
never runs on it. `memref.subview` is worse (its stride inference asserts on the
`#ttcore.shard` layout, failing at bufferize). The `#shard` operand is simply not
memref-viewable, and `remote_load` only avoids this by COPYING into a plain #l1
shard buffer.

**Corrected fix (no view op):** `fabric_recv`'s bufferize must NOT emit any memref
view (subview/reinterpret_cast). Instead the op produces the shard via its OWN
result -- a memref-result form `%shard = d2m.fabric_recv %recvBuf[%i] : memref<shard>`
whose result aliases the recv operand's allocation (getAliasingValues already says
so). split-v2 then maps that result to `getOrCreateCB(recv)` and the
reserve+push lowering exposes the fabric-written CB; the compute reads `%shard`,
rewired to the CB wait result. Concretely this needs:
1. a memref-result builder/form on `FabricRecvOp` (no localBuffer, result = shard
   memref);
2. bufferize emits that form + `ToTensorOp(result)` (no subview/reinterpret_cast);
3. `traceComputeMemrefToCB` recognizes a memref defined by `FabricRecvOp` and
   returns its recv operand (so compute consumption is tied to recv's CB);
4. confirm the pre-split be-pipeline passes (dst-register-access, linalg-to-affine,
   generic-linearize-memref) pass through `fabric_recv` (it is a ShardDMAOpInterface
   DM op, so they should skip it -- verify).

The grid-[N,1] dynamic recv-slot offset is still a separate follow-up after 3a.

---

## Milestone 3a — PIVOT to the tmp-scratch approach (2026-06-24, locked)

The `#shard` recv operand is fundamentally not memref-viewable, and reading it
back over NoC contends with the open fabric connection. Both problems vanish if
the recv buffer is a plain gridless `#l1` scratch instead of a `#shard` operand.

### De-risk: symmetric scratch addressing is correct *by construction*

The fabric-write dst address is computed in `D2MDMAWriteRewriter`
(`D2MToTTKernel.cpp`) via `buildNocEndpoint(dst, dstIndices, DeviceL1)` =
`castCBTypeAsAddress(dst_CB) + offset` at the dst-core's virtual coords. All
devices run the same SPMD kernel, so a given scratch CB is allocated at the SAME
L1 offset on every device. Therefore the sender's `castCBTypeAsAddress(scratchCB)`
already equals the peer's address for that same scratch — symmetric addressing
needs no probe. (No device run required to confirm; it falls straight out of the
CB→address lowering + SPMD allocation.)

### The shape that makes both sides trivial

- **tmp scratch**: an in-kernel `d2m.empty(<shard>)` → gridless `#l1`,
  shard-shaped. The recv consumes it *view-free* (it IS already the shard).
- **recv** `fabric_recv(tmp)` with EMPTY indices: bufferize → `ToTensorOp(tmp)`
  directly (skip the reinterpret_cast entirely when gridRank==0). No memref view
  ⇒ no `memref.reinterpret_cast` ⇒ no legalization failure. Lowering unchanged
  (reserve+push on tmp's CB, no dma_read).
- **send** = `remote_store` learning a **scratch-dst mode**, NOT a new op
  (remote_store already has all the bufferize/split/DPS machinery):
  - detect scratch-dst by `!hasDeviceLayout(dst) && !startDevice.empty()`;
  - relax `RemoteStoreOp::verify` to allow that case (skip device-layout +
    grid-rank checks);
  - in `LowerLoadStoreOpsToDMA`, for scratch-dst emit a **fully-indexed**
    `DMAWriteOp(src, scratch, dstIndices=[0,0,0], numElems=shardVolume,
    startDevice, deviceMcastShape)` (NOT the shard-level form) + `SemaphoreInc`.
    `[0,0,0]` ⇒ `buildNocEndpoint` returns `castCBTypeAsAddress(scratchCB)` at
    core (0,0) = the peer's symmetric scratch. Reuses the existing
    `D2MDMAWriteRewriter` fabric path with ZERO new D2MToTTKernel code and no
    memoryMap dependency (we supply the 3 indices directly).

This avoids a whole new op (the earlier "dedicated fabric_send" idea) — the only
genuinely new op stays `fabric_recv`, now repointed at the gridless `#l1` scratch.

### Ring step (target shape)
```
tmp = empty(<shard>)                 # gridless #l1 scratch
remote_store(tmp, [], partial, start_device=[dy,nbr], device_mcast_shape=[1,1],
             semaphore=es, semaphore_indices=[cy,0])   # write peer's tmp
semaphore_wait(es, 1)
s = fabric_recv(tmp, []) + remote_load(in0, [idx])     # view-free consume
```

### Edits
1. `RemoteStoreOp::verify` — allow gridless `#l1` dst when `startDevice` present.
2. `LowerLoadStoreOpsToDMA` `D2MLowerRemoteStoreRewritePattern` — scratch-dst ⇒
   fully-indexed `dma_write([0,0,0], numElems=vol)`.
3. `FabricRecvOp::verify` — allow gridless `#l1` operand (empty indices).
4. `FabricRecvOp::bufferize` — gridRank==0 ⇒ `ToTensorOp(buffer)` directly (no view).
5. Confirm split-v2 gives the scratch (remote_store DST + fabric_recv operand) a CB.
6. `_m3a.py` — rewrite to the tmp-scratch form above.

Follow-up after 3a: grid-[N,1] dynamic recv-slot offset (still hardcoded 0).

### Milestone 3a — DONE (2026-06-24): tmp-scratch fabric_recv PASSES on device

`test/d2m-jit/_m3a.py` PASSES on the 4-chip Blackhole ring (all devices
maxdiff ~0.005). The cross-device fabric write into a peer's gridless #l1 scratch
+ the view-free `fabric_recv` consume (NO noc_async_read) both work on silicon.
Existing #shard-path all_gather tests still pass (changes are gated to the new
scratch path). Implemented edits (committed):
1. `RemoteStoreOp::verify` — scratch-dst branch (gridless #l1 dst + no grid
   indices + skip operand-reference check when startDevice present).
2. `DMAWriteOp::verify` — a cross-device write with a local-layout dst (fabric
   scratch) requires 3 dst indices like a remote dst.
3. `LowerLoadStoreOpsToDMA` — scratch-dst remote_store -> fully-indexed
   `dma_write(srcIdx=[0], dstIdx=[0,0,0], numElems=vol)`.
4. `D2MDMAWriteRewriter` (D2MToTTKernel) — fabric write to a LOCAL #l1 scratch
   dst uses `get_write_ptr(scratchCB)` for the base address (NOT
   castCBTypeAsAddress, which only resolves for remote/compile-time operands);
   coords from dstIndices[0,1], offset dstIndices[2].
5. `FabricRecvOp::verify`/`bufferize` — gridless #l1 scratch operand: no grid
   indices, and bufferize exposes the buffer directly (no reinterpret_cast).

Next: 3b (2-step ring that hung before -> should now be hang-free since the recv
is view-free and never reads back over NoC), then 3c (full reduce_scatter ->
all_reduce). Follow-up: grid-[N,1] dynamic recv-slot offset (still 0).

### Milestone 3b — HANGS (2026-06-24): compute-thread mis-scheduling deadlock

`test/d2m-jit/_m3b.py` (2-step chained ring, view-free fabric_recv, separate sems
es0/es1) deadlocks on device though it compiles. The read-back is gone, so it was
not the sole cause. NEW root cause (from `python _m3b.py dump`): the COMPUTE thread
of the unrolled 2-step ring is mis-ordered --
  wait(es0); wait(tmp0); reserve(acc1); wait(es1); wait(tmp1); reserve(acc2);
  ...; push(acc1); push(acc2)
It waits es1 BEFORE pushing acc1. The DM thread needs acc1 (cb_wait_front) to send1
-> inc peer's es1. Circular across the ring -> deadlock. The compute scheduler
hoisted both steps' reserves/waits ahead of both steps' pushes, breaking the ring's
per-step producer ordering (step t push must precede step t+1 recv-wait). The DM
fabric thread itself is correctly ordered. FIX: keep per-step compute order
(push acc_t before step t+1's semaphore_wait/recv reserve). Detailed in memory
[[d2m-ring-interleaved-fabric-hang]]. Then 3c.

### Milestone 3b — FIXED (2026-06-25): loop-fusion barrier on semaphore_wait

Root cause of the 2-step deadlock: `D2MInsertSpillAndScratch::fuseOuterScfLoops`
fuses the two structurally-identical compute nests (acc1, acc2) and hoists the ops
sitting BETWEEN them -- including the ring's `semaphore_wait(es1)` -- to before the
first nest (as "setup" ops). After thread-split, the compute thread then waits on
es1 before pushing acc1; since es1 is only satisfied by a peer that first consumed
acc1, the ring deadlocks. (DM thread was correctly ordered; CB handshakes cover its
hoisted send.)

Fix: in `fuseOuterScfLoops`, after sorting chains into source order, bail out of
fusion (`return false`) if a `d2m.semaphore_wait` sits between consecutive nests.
A consume-side wait carries cross-thread/cross-device ordering; a produce-side
`semaphore_inc`/`set` (fused all_gather's cross-device store signalling a peer) does
NOT and those kernels rely on fusion -- so the barrier is `SemaphoreWaitOp` only.
The ring nests write distinct buffers and need no fusion, so skipping it is correct.

After the fix the compute thread is correctly ordered (push acc1 BEFORE wait es1,
and the recv CBs get their cb_pop_front). `_m3b.py` PASSES on the 1x4 Blackhole ring
(all devices maxdiff ~0.007, no hang). Fusion-heavy matmul all_gather tests
(chunked, streaming) still pass; the remaining suite failures are the environmental
fabric-router-sync bringup timeout (reproduces on baseline without this change).

Next: 3c -- the full (N-1)-step reduce_scatter, then + all_gather = all_reduce.

### Milestone 3c — DONE (2026-06-25): full ring all_reduce PASSES on device

`test/d2m-jit/_m3c.py` PASSES on the 1x4 Blackhole ring: a 3-step (N-1) chained
ring all_reduce where every device ends with the identical full sum
(sum over all devices), maxdiff ~0.01, no hang. This composes all the milestone-3
pieces: tmp-scratch `fabric_recv` (3a, view-free, no NoC read-back), the
fusion-barrier fix (3b, no compute-thread deadlock), and:

**Send-only forwarding (new constraint learned).** A value that is fabric-sent
must be send-only -- a CB consumed by BOTH a DM `remote_store` and a compute op
fails to legalize ("unresolved materialization memref->cb ... live user"). The
circulate-and-accumulate ring inherently wants to forward each received value
(needed by compute for the accumulate too). Worked around by making every
forwarded value a send-only compute output:
- load in0[0] TWICE (a send-only copy `v_send` for step 1 + a compute copy `v`);
- forward each received r_k as `acc_k - acc_{k-1}` (== r_k) -- a send-only compute
  output -- so the raw recv result stays single-(compute-)consumer.

This is correct but NOT bandwidth-optimal (sends the whole shard each step, O(N)
data vs the chunked reduce-scatter's O(1)). Follow-ups:
1. Bandwidth-optimal chunked reduce-scatter + all-gather (each step moves 1/N).
2. Compiler support for a recv buffer consumed by both compute and a DM forward
   (materialize once), which would remove the double-load / subtraction workaround.
3. grid-[N,1] dynamic recv-slot offset (still hardcoded 0).

### Milestone 3d — DONE (2026-06-25): bandwidth-optimal CHUNKED ring all_reduce

`test/d2m-jit/_m3d.py` PASSES on the 1x4 Blackhole ring: full ring reduce-scatter
(N-1 steps) + all-gather (N-1 steps), moving ONE chunk per step (O(1) vs _m3c's
O(N) whole-vector). Every device ends with the identical full sum, maxdiff ~0.01,
no hang. Three techniques made it work:

1. **p-relative chunk indexing.** The ring's per-step chunk indices depend on the
   runtime device id p. Track chunks in p-RELATIVE order (`c[r]`, r=0..N-1) so the
   per-step send/accumulate logic uses CONSTANT rel-indices (unrollable in Python);
   only the operand load/store use the runtime actual index `(p+r)%N`
   (`remote_load`/`remote_store` accept dynamic indices). Send rel `-k`, accumulate
   into rel `-k-1` (reduce-scatter); send rel `1-k`, place into rel `-k`
   (all-gather). Send-from-peer lands in the same ACTUAL chunk on the neighbor, so
   the reduction is correct.

2. **Reduce-scatter is naturally clean.** Each forwarded chunk is either an
   original load (step 0) or the previous step's accumulate output, and is
   send-only at its step (the step's add targets a different chunk). No copies.

3. **All-gather send/store split.** A reduced chunk must be both FORWARDED (fabric
   send) and kept for the final local store, but a value cannot feed two DM ops --
   the first pops the CB and the second's `cb_wait_front` hangs (this was the
   initial _m3d deadlock). Fix: for each such chunk make TWO independent
   add-to-zero compute copies (a send-only one + a store-only one) from the raw
   recv; the recv stays a compute-only consumer. Needs a `zeros` operand for the
   add-to-zero placement copies.

Follow-ups: (1) compiler support for a recv buffer feeding both compute and a DM
forward (would remove the double-load / copy workarounds in _m3c/_m3d); (2)
grid-[N,1] dynamic recv-slot offset (still 0); (3) generalize N / chunk size and
fold into a reusable `all_reduce` DSL primitive.

### Milestone 3e — DONE (2026-06-25): GENERIC, loop-written ring all_reduce + `static_range`

`test/d2m-jit/_m3e.py` PASSES on the 1x4 Blackhole ring: a ring all_reduce written
with a `for k in static_range(N-1)` loop and parameterized over the mesh volume N
(a closed-over int capture) instead of hand-unrolling for N=4.

New DSL feature `static_range` (lib `_src/ast.py` `visit_For`): a loop marker that
UNROLLS the body at trace time (Python-level), with bounds evaluated from int
literals / int captures (`_eval_static_int`). Regular `range()` still lowers to
`scf.for`. This is additive -- existing range/scf.for loops are unaffected
(streaming/chunked matmul regression pass).

Why unroll rather than a real runtime `scf.for`: a runtime loop carrying the ring's
cross-iteration state is blocked two ways --
1. loop-carried *tensor* iter_args fail one-shot-bufferize ("Yield operand not
   equivalent to iter bbArg": each tile op yields a fresh buffer != the iter_arg);
2. keeping state in an operand (RMW in the loop) instead: the runtime loop forms
   (`scf.for` with fabric ops inside -- verified), but the DSL re-allocs the
   operand per store and reading a grid OUTPUT operand back demotes it to #l1,
   where a local `remote_store` is rejected ("must be remote").
`static_range` sidesteps both: unrolled, the accumulators are plain SSA
reassignments (like _m3c/_m3d) and there is no loop-carried iter_arg / operand RMW.

Algorithm = circulate-and-accumulate (O(N) bandwidth, SSA state): `acc`, `acc_prev`
reassigned each step; forward `acc - acc_prev` (== last recv, send-only compute
output), then `acc += r`. `acc_prev` is seeded from an opaque ZEROS operand (NOT
`v - v`, which folds to a literal zero -> `acc - acc_prev` canonicalizes to `v` and
sends `v` directly while compute also reads it -> illegal compute+DM CB share).
Cumulative semaphore counts (`wait(es, k+1)`) now work -- the earlier "cumulative
hangs" was the read-back/fusion confound, fixed in 3a/3b.

Genericity: the algorithm (steps, indices `(p+1)%N`, `static_range(N-1)`,
`mcast_shape=[1,N]`) is generic over N; only `device_synchronize`'s `num_receivers`
stays a literal (it lowers to an i32 attribute; the kwargs_as_attr lambda sees only
the AST node, not the int captures -- folding captured constants there is a small
follow-up). The bandwidth-optimal CHUNKED ring can't be loop-written this way: it
needs runtime-indexed chunk slots (operand RMW -> the #l1 wall above; SSA chunk
vars can't be indexed by a loop var in the tracer), so the chunked variant stays
the unrolled _m3d.

### Runtime-loop (scf.for) investigation — in-place accumulate (2026-06-25)

Tried to enable a TRUE runtime `for k in range(N-1)` ring (not `static_range`
unroll) via an in-place eltwise accumulate, the eltwise dual of `c += a@b`:
`__add_acc__` (linalg `outs(acc)`), `copy_` (in-place replace via `dst += src-dst`
since a pure identity-copy linalg gets canonicalized away), and a `visit_AugAssign`
branch routing `acc += <eltwise>` to it. Findings (all on the 1x4 Blackhole):

- It DOES fix the bufferization error: `acc += r` makes a loop-carried accumulator
  yield a buffer equivalent to its iter_arg, so the minimal runtime loop COMPILES
  (where `acc = acc + r` failed "Yield operand not equivalent to iter bbArg").
- Single (non-loop) in-place `+=` computes CORRECTLY (maxdiff ~0.005).
- BUT a LOOP-CARRIED in-place accumulate MISCOMPUTES: it runs without hanging but
  `acc` doesn't accumulate across iterations (stays ~initial). matmul `c += a@b`
  works in a loop because tile_matmul accumulates through the DST/L1-acc path
  (cf. [[chunked-matmul-l1acc-loop-bug]]); plain eltwise has no equivalent, so the
  loop-carried buffer isn't threaded at runtime. This is a silent wrong-answer, so
  auto-routing `+=` to in-place is worse than the loud bufferization error.
- Fabric ops inside an scf.for: compile (with a single in-place carry) but HANG on
  device.

Conclusion: a real runtime-loop ring needs compiler work on BOTH (a) loop-carried
eltwise accumulate lowering (DST/L1-acc threading, like the matmul path) and (b)
fabric-ops-in-scf.for runtime support. The experimental DSL ops were reverted (not
landed) since they silently miscompute in loops. The `static_range` unroll
(milestone 3e) remains the working generic-over-N, loop-written approach.

### Root cause of the loop-carried eltwise miscompute (2026-06-25)

Dumping the final ttkernel compute kernel for a minimal `acc += r` runtime loop
pins it exactly. The accumulator is lowered to round-trip through its output CB
every iteration, with DST acquired/released INSIDE the loop:
```
cb_reserve_back(%acc)                 # reserved ONCE, before the loop
scf.for k {
  cb_wait_front(%in)
  tile_regs_acquire()                 # DST per-iteration
  copy_tile(%acc,0,0)                 # DST0 <- acc CB (read)
  copy_tile(%in,0,1); add_binary_tile(0,1,0)
  pack_tile(0,%acc,0)                 # acc CB <- DST0 (write)
  tile_regs_release(); cb_pop_front(%in)
}
cb_push_back(%acc)                    # pushed ONCE, after the loop
```
The acc CB is `reserve`d once / `push`ed once with NO per-iteration push/pop and NO
`cb_wait_front`. So (1) `copy_tile(%acc)` reads the CB front, which the in-loop
`pack_tile` (writing the reserved back) never updates without cycling the CB -> the
carry doesn't propagate (iter k doesn't see iter k-1's result); and (2) the initial
value (`acc = remote_load(zin)`) is never threaded into %acc (no wait_front before
the loop). Both are silent -- the IR is well-formed and bufferization is satisfied
(the outs(acc) aliasing is locally valid), so nothing flags it.

matmul `c += a@b` works in a loop because it keeps the accumulator RESIDENT in DST
across the whole reduction (tile_regs_acquire before the loop, accumulate in DST
each iter via the L1-acc/DST path, pack once after) -- no per-iteration CB
round-trip. Eltwise has no such DST-resident-across-loop accumulation.

Fix: the compute lowering (InsertDstRegisterAccess + CB-sync scheduling) must keep
a loop-carried eltwise accumulator in DST across the loop body (or give the acc CB
a correct per-iteration read<->write handshake + load its init), i.e. extend the
matmul DST/L1-acc treatment to eltwise. A real C++ change, not a DSL fix.

### Root cause CORRECTED (2026-06-25): it's the init handshake, not the carry

The earlier "carry doesn't propagate" framing was wrong. For a single-tile
accumulator the CB read/write pointers coincide at the base slot, so keeping them
INVARIANT over the loop (no per-iteration push/pop) is correct -- the in-place
add accumulates fine. The real bug is INITIALIZATION, confirmed in the post-split
(be-pipeline) IR of the accumulate generic:
- `datamovement_kernel6` (DM): reserve(accCB); dma_read zin -> accCB (the ZEROS);
  push(accCB) -- the init IS loaded and pushed.
- `compute_kernel8` (compute): **`d2m.reserve(accCB)`** before the loop, then the
  in-place accumulate loop, then push(accCB).

The compute opens the accumulator CB with `reserve` (producer/write side -> fresh
uninitialized slot) instead of `wait` (consumer/read side -> the zeros the DM
thread just pushed). So the pushed init is never consumed; iter 0 reads an
uninitialized slot and the result is `garbage + N*input` instead of `0 + N*input`.

Why: the accumulator is BOTH a consumer (of the DM-pushed init) AND the output
(compute produces, DM later stores), but split-v2's compute-side CB classification
is binary input(wait/pop) XOR output(reserve/push). Because compute WRITES the
accumulator (the in-place add's outs(acc)), it's classed output-only -> reserve+
push, dropping the init-consuming wait.

Fix: teach split-v2's CB-sync that a loop-carried, externally-initialized in-place
accumulator is a consume-then-reproduce CB: emit `wait(accCB)` (consume the init)
ONCE before the loop, the invariant-pointer in-place loop, then `push(accCB)` once
after -- i.e. wait, NOT reserve, before the loop. Supersedes the earlier
"DST-resident/CB-round-trip" diagnosis above.

### split-v2 wait-fix: implemented + tried (2026-06-25) — necessary but NOT sufficient

Implemented the split-v2 fix (in insertComputeCBOpsV2): an `initAccumulator`
(outputCB && info.consumed && info.dmLoad) acquires with `WaitOp` (consume the
DM-pushed init) instead of `ReserveOp`. IR confirms compute_kernel now does
`cb_wait_front(acc)` before the loop. Device results:
- 1-iteration loop: PASSES (init consumed -> got = 0 + 1*input). So the init bug
  IS fixed by the wait.
- N-iteration loop: still FAILS. The accumulator round-trips through its CB every
  iteration (copy_tile reads FRONT, pack_tile writes BACK) and the CB is
  double-buffered (#ttcore.cb_layout<...,2>), so front != back -> each iteration
  re-reads the stale init front instead of the prior result. The cross-iteration
  carry is lost.

So BOTH fixes are needed: (1) split-v2 wait (init-consume, done & confirmed) AND
(2) keep the eltwise accumulator RESIDENT in DST across the loop (load init into
DST once before, accumulate in DST each iter, pack once after) so it doesn't
round-trip the CB -- the matmul L1-acc treatment, currently gated to matmul in
InsertDstRegisterAccess (hasTileMatmul / d2m.reduction_loop / packer-L1-acc). That
is a substantial change to a 2.4k-line pass. The split-v2 change alone leaves
multi-iter eltwise accumulate SILENTLY wrong and could also affect matmul
accumulators (same outputCB+consumed+dmLoad shape), so it was reverted. The
`static_range` unroll (3e) remains the working generic-over-N approach. A true
runtime-loop ring needs the InsertDstRegisterAccess DST-resident-eltwise work
(plus the earlier fabric-ops-in-scf.for runtime support).

### Race hypothesis RULED OUT (2026-06-25): it's CB read/write pointer divergence

Checked whether the multi-iter carry failure was a pack->unpack race: extended
GenericTileComputeLoops to insert `unpack_stall_on_pack` for a self-RMW linalg
(ins aliases outs, e.g. `acc += x`). The stall IS now emitted before the in-loop
`copy_tile(acc)` -- and the multi-iteration loop STILL fails. So it is NOT a race.

The real mechanism: after the DM pushes the init, `cb_wait_front` puts the READ
pointer at the init page W, so `copy_tile(acc)` reads page W; but the push advanced
the WRITE pointer to W+1, so `pack_tile(acc)` writes page W+1. Read and write target
DIFFERENT pages, so every iteration re-reads the init page W -- a CB round-trip
(read-front / write-back) fundamentally cannot do in-place read-modify-write, and a
stall can't fix a different-page read/write. (Combined with the split-v2 wait-fix
the 1-iteration loop passes -- one step needs no carry.)

Therefore the ONLY correct carry fix is DST-resident accumulation (the matmul
path): load the accumulator into DST once BEFORE the loop, accumulate in DST each
iteration, pack once AFTER -- no per-iteration copy_tile/pack_tile of the
accumulator. This is an InsertDstRegisterAccess change (generalize its
loop-resident-DST-accumulator handling, currently gated to matmul via
hasTileMatmul / d2m.reduction_loop, to a loop-carried eltwise accumulator). Plus
the split-v2 wait-fix for the init. All experiments reverted; static_range unroll
remains the working generic approach.

### DEFINITIVE root cause via matmul contrast (2026-06-25)

(Pointers do NOT move in the loop -- correct.) Dumped a matmul-acc user loop
(`acc = a@b; for k: acc += a@b`) and compared its compute kernel to the eltwise one:

matmul loop body (works):
```
pack_reconfig_l1_acc(%k != first)     # packer L1-acc: off iter0 (overwrite), on after
matmul_block(a, b) -> DST
pack_tile(%acc)                        # packer adds DST into %acc IN PLACE
```
NO copy_tile(%acc) -- the accumulator is never read back; the running sum lives in
the CB and is accumulated by the hardware packer.

eltwise loop body (broken):
```
copy_tile(%acc) -> DST0                # READS the accumulator back
copy_tile(%in)  -> DST1
add_binary_tile -> DST0
pack_tile(%acc)                        # overwrite (no pack_reconfig_l1_acc)
```
copy_tile reads the CB READ pointer (front = the DM-pushed init page); pack_tile
writes the WRITE pointer (a different page). Both fixed across the loop, so every
iteration re-reads the init -> %acc = init + input (overwritten), store reads the
init page -> got ~= init. The earlier "carry/race/divergence-in-loop" framings were
imprecise: the real issue is that eltwise READS the accumulator back at all.

FIX: lower a loop-carried eltwise accumulate like matmul -- drop the copy_tile(acc)
read; DST = input; pack_reconfig_l1_acc(k != first) + pack_tile(acc) so the packer
does acc += input in place. This is the packer-L1-acc path, gated to matmul in
InsertDstRegisterAccess (Scheduled.cpp disablePackerL1Acc = ...||!hasTileMatmul||...
+ allTileMatmulOutputsSupportPackerL1Acc); extend it to recognize a loop-carried
eltwise add accumulator. Bonus: first-iter-overwrite makes the explicit zeros-init
and the split-v2 wait-fix unnecessary (matmul needs neither). Supersedes the prior
diagnoses in this doc.

### CORRECTION (2026-06-25): it's user-loop (scf.for iter_arg) accumulation, not eltwise/copy_tile

Per the suggestion, dumped+RAN the matmul accumulator loop with L1-acc DISABLED.
disable-l1-acc forces the copy_tile-reload path (copy_tile(acc) + matmul_block +
pack_tile(acc)) -- structurally identical to the eltwise loop. Device results for a
USER-loop accumulator (`acc = a@b; for k: acc += a@b`, N=4):
- matmul, L1-acc DISABLED (copy_tile-reload): FAIL (maxdiff ~19)
- matmul, L1-acc ENABLED (packer):            FAIL (maxdiff ~20)
- eltwise (copy_tile-reload):                 FAIL (maxdiff ~10)

So the bug is NOT copy_tile-vs-packer and NOT eltwise-specific: a LOOP-CARRIED
ACCUMULATOR IN A USER scf.for is broken for BOTH ops and BOTH L1-acc modes. (And
the 1-iteration loop passes for all -- so it's specifically the cross-iteration
carry of an scf.for iter_arg accumulator.) The matmul tests that PASS use the
matmul block's INTERNAL K-reduction, which is not a user scf.for iter_arg.

This supersedes the earlier copy_tile / packer-L1-acc / read-write-pointer
diagnoses. The real blocker for a runtime-loop ring is correct lowering of an
scf.for iter_arg accumulator (the CB handshake / DST handling across iterations),
affecting any user-loop accumulation regardless of the compute op. static_range
unroll (which has no iter_arg) remains the only working path.

### COMPLETE root cause (2026-06-25): scf.for iter_arg accumulator -> FIFO page-split

Confirmed the user-loop accumulator is broken for ALL variants (eltwise, matmul
copy_tile-reload, matmul packer-l1-acc, peeled AND non-peeled). Dug into the
lowering:

- The L1-acc / DST-resident accumulation in InsertDstRegisterAccess is gated to the
  matmul's `d2m.reduction_loop` / `d2m.blocking_loop` (Shared.cpp
  isReductionBlockingLoop). A USER scf.for iter_arg accumulator has neither attr, so
  it takes the GENERIC per-iteration path: tile_regs_acquire / copy_tile(acc) /
  compute / pack_tile(acc) / tile_regs_release INSIDE the loop, round-tripping the
  accumulator through its CB every iteration.
- The accumulator CB is then used as a FIFO across threads: the DM pushes the init
  (acc = remote_load(zeros)) to page0, compute packs the result to page1 (the push
  advanced the write ptr past page0), and the DM store does cb_wait_front -> reads
  page0 (the init, first in FIFO), not page1. => got ~= init. The standard matmul
  reduction works because the K-reduction lives INSIDE matmul_block (no compute-level
  scf.for accumulator, no per-iteration CB round-trip, one pack -> one FIFO entry).

FIX (substantial): give a loop-carried scf.for iter_arg accumulator the
matmul-reduction treatment -- keep it DST-resident across the loop (tile_regs_acquire
BEFORE the loop, init in-register on the first iteration, accumulate in DST inside,
ONE pack AFTER the loop), and DROP the DM init-load (no separate FIFO entry). That is
a real InsertDstRegisterAccess feature (recognize the iter_arg accumulator, hoist DST
acquire/pack out of the loop), plus matching CB-handshake changes -- not a localized
fix, and it must not regress the matmul reduction_loop path. static_range unroll
(no iter_arg accumulator -> no per-iteration CB round-trip) remains the working path
and is what the shipped all_reduce kernels (_m3c/_m3d/_m3e) use.

### Single-buffered CB: tried, does NOT fix it (2026-06-25)

Forced all synchronized CBs to single-buffered (HoistCBAllocs: attach
CBLayoutAttr numBuffers=1; confirmed all cb_layout became <4096x4096, 1>), with the
in-place op + split-v2 wait-fix re-applied. The multi-iteration eltwise loop STILL
fails (maxdiff ~10, got ~= init), and does NOT deadlock. So CB buffer depth is not
the lever -- the per-iteration CB round-trip of a loop-carried accumulator
(copy_tile read / pack_tile write across the thread-crossing accumulator CB) is
broken for both single- and double-buffered CBs. Reinforces that the fix must avoid
the per-iteration CB round-trip entirely (DST-resident accumulation, the
matmul-reduction treatment), not tune the CB depth. Reverted.

### BREAKTHROUGH (2026-06-26): runtime-loop ring works — the init must be COMPUTE-owned

The whole "scf.for iter_arg accumulator is broken" diagnosis above had ONE missing
piece: the accumulator was being **DM-seeded** (`acc = remote_load(zeros)`), which is
exactly what put the init on a separate CB FIFO page from the compute pack -> the
page-split miscompute. The fix is NOT a new InsertDstRegisterAccess feature; it is to
make the accumulator **compute-initialized**:

```
acc = zeros([M, K])      # compute-owned init (a TileFillOp), NOT remote_load
own = empty([M, K]); remote_load(own, in0, [cy, cx])
acc += own               # __add_acc__  (linalg outs(acc))
acc_prev = zeros([M, K])
for k in range(N - 1):   # GENUINE runtime scf.for
    fwd = acc - acc_prev
    remote_store(t, [], fwd, ...); semaphore_wait(es, k + 1)
    r = fabric_recv(t, [])
    acc_prev = copy_(acc_prev, acc)
    acc += r
remote_store(out, [cy, cx], acc)
```

A compute-owned `zeros` init and the compute pack land on the SAME CB page, so the
loop-carried iter_arg threads correctly. Two supporting fixes landed with it:
- a **DM-seed guard** in `visit_AugAssign`/`visit_For` (ast.py): `acc = remote_load(...)`
  / `fabric_recv(...)` followed by an in-loop `acc += ...` now raises a clear
  D2mJitError instead of silently miscompiling (the page-split trap above);
- `MarkSynchronizedBuffers` no longer tags the zeros-fill (TileFillOp producer) as a
  dead `compute_intermediate`, so Allocate/HoistCBAllocs keep the init buffer live
  (wall #1).

Shipped as `test/d2m-jit/test_ring_all_reduce_loop.py` (single-core) and
`test_all_reduce_grid.py` (multi-core grid, with the my_logical scratch-fabric-write
fix). Both PASS on the 1x4 Blackhole.

### `static_range` REMOVED (2026-06-26)

With the runtime `range()` ring working, the trace-time-unroll `static_range` marker
is no longer needed and has been **deleted** from the DSL (`_src/ast.py` `visit_For`
now accepts only `range`; `_eval_static_int` stays — it is still used for compile-time
attrs like shape literals / `num_receivers`). `_m3e.py` was migrated to `for k in
range(N-1)` (PASSES). The bandwidth-optimal chunked ring `_m3d` stays hand-unrolled
(it needs runtime-indexed chunk slots, an operand-RMW pattern the runtime loop still
can't express); it never used `static_range`, so the removal does not affect it. All
the milestone-3e-and-later "static_range unroll remains the only working path"
conclusions above are SUPERSEDED by the runtime-loop breakthrough.
