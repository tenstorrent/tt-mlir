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
