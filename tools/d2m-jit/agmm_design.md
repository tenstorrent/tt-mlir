# Porting all_gather_minimal_matmul_async to d2m-jit — design + status

Goal: port the concept of
`third_party/tt-metal/.../models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py`
into d2m-jit, starting small and scaling up.

## The reference op (what it computes)

`ttnn.experimental.all_gather_minimal_matmul_async(input, weight, ...)` is a
**fused all_gather + matmul** for tensor/sequence-parallel linears (Wan2.2 DiT
qkv / dense-attn / ff1 layers). On a 2D mesh (`sp_axis`, `tp_axis`):

- `input` (activations `[M, K]`) is **sequence-parallel** sharded along M: device
  has `[M/sp, K]`.
- **all_gather over the SP axis** -> every device holds the full `[M, K]`.
- `weight` (`[K, N]`) is **tensor-parallel** sharded along N: device has `[K, N/tp]`.
- **matmul**: gathered `[M, K] @ [K, N/tp] = [M, N/tp]` per device.
- Fused/async: the all_gather is overlapped with the matmul (start matmul on
  chunks as they arrive); plus optional bias / gelu / addcmul / chunked-N.

Reference torch (test line 146): `torch_output = torch_input @ weight_input`.
Non-fused path also exists: `all_gather_async` then `minimal_matmul` as two ops.
Test shapes are large (M up to 115200, K=5120, N up to 4096) for a (2,4)/(4,8)
cluster.

## d2m-jit port plan (1D, this 1x4 Blackhole)

The box is a 1x4 ring (1D mesh), so the port is a 1D specialization: all_gather
activations over the 4-device ring, matmul with a weight. With no separate TP
axis the weight is **replicated** (each device computes the full `[M, N]` =
`gathered @ weight`); a real TP split needs a 2D mesh.

Start small ([M,K,N] a few tiles), scale up.

## Status

WORKING END TO END (small): `scratchpad/agmm4.py` -- all_gather (sp, 1x4) ->
TP-sharded matmul -> per-device N-slice -> mesh_gather to full `[M, N]`. PASS
(M=128, K=64, N=256, rel 0.0017). KEY INSIGHT: the weight must be **TP-sharded**
along N (`mesh_shard(W, shard_dims=[0,1], shard_shape=[1,N])`, each device its
N-slice -- the real AGMM form), NOT replicated. Replication is what failed:
`to_layout` (single-device) on a mesh -> buffer-size assert; `mesh_shard
shard_shape=[1,1]` -> compiler crash ("incorrect fold result type"). With a
TP-sharded weight (sharded operands, like the ring kernels) there is no
replication and the AG-kernel -> MM-kernel chaining works. The MM is row-parallel
(grid (M_tiles,1)); each core does `gathered_row @ weight_slice`; the activation
(gathered, per-device-full from the AG) and weight (TP shard) are both sharded.

(superseded) early notes:
- **Step 1 — 1x4 all_gather** (`scratchpad/agmm1.py`): each device mcasts its
  shard to all devices' gathered buffer (`remote_store(out,[dx,0],buf,
  device_mcast_shape=[1,N], semaphore=es)`, then `semaphore_wait(es,N-1)`); every
  device ends with the full gathered `[M,K]`. PASS (shape 128x128, rel 0.002).
  This is the all_gather building block (the existing test_ccl_all_gather is 1x8;
  this is the 1x4 single-core form). Gather layout: out buffer is `[1,K]` row
  blocks on an `[N,1]` grid (one block per device's row).

BLOCKERS (the matmul half):
- **matmul on a multi-device mesh fails** at runtime:
  `LOG_ASSERT meshBuffer.size() == tensorDesc.sizeBytes()`
  (runtime/lib/ttmetal/executor_utils.h:384). Repro `scratchpad/mmonly.py`: a
  row-parallel matmul (`grid=(N,1)`, each core `gathered_row[cy] @ weight`) on
  `d2m.mesh((1,4))` with `to_layout`-replicated operands. The existing matmul
  tests all use single-device `d2m.mesh((1,1))`; distributing/replicating matmul
  operands across a multi-device mesh has a buffer-descriptor mismatch that needs
  sorting (likely the replicated `to_layout` operand or the `mesh_gather` of the
  output on a `[1,N]` shard_shape). THIS is the first thing to fix to get a
  non-fused AG+MM.
- **on-device kernel chaining** (AG kernel output -> MM kernel input) hits the
  same buffer-size assert (`scratchpad/agmm2.py`/`agmm3.py`) -- but since MM-only
  fails too, the root cause is the multi-device-mesh matmul, not composition per
  se. Re-test chaining once the mesh matmul works.

## Fusing AG+MM into ONE kernel — attempted, blocked at framework level

NOTE (2026-06-26): the `static_range` DSL marker used below has since been REMOVED
(runtime `range()` loops now work — see all_reduce_design.md "BREAKTHROUGH"). The
fused-kernel investigation below predates that and was written against the
trace-time unroll. RETRIED with a runtime `for r in range(N)` (one matmul generic
with an scf.for, `scratchpad/agf_range.py`): it does NOT sidestep the ordering
issue — `rel=1.37`, WORSE than the unrolled `rel=0.54`. The matmul's
`remote_load(g)` still races the DM fabric writes with no compute<->DM ordering
(the runtime-indexed `g[r]` read is unordered w.r.t. the AG's fabric writes). So
the conclusion stands regardless of unroll-vs-runtime-loop: fusion needs the
framework fix below.

`scratchpad/agf.py` — single-core fused kernel: AG mcasts the shard to all
devices' gathered `g` (`[N,1]` row-block grid), `semaphore_wait(N-1)`, then a
`range(N)` loop matmuls each gathered row `g[r] @ weight_slice` and stores
`out[r]`. Findings:

- **matmul reads `g` via `remote_load`**: COMPILES + RUNS (no hang/crash) but
  WRONG (rel ~0.54). Two root issues: (a) no compute<->DM barrier within one
  kernel -- the compute thread's `remote_load(g)` races the DM thread's fabric
  writes (`semaphore_wait` is DM-side; compute doesn't wait for it); (b) `g` is
  both the AG output (remote_store/DM) and the matmul input (remote_load/DM-read)
  -- split-v2 likely allocates these as DIFFERENT buffers, so the matmul reads an
  unwritten `g`. The non-fused version is correct only because the AG kernel fully
  completes (program boundary) before the MM kernel.
- **matmul reads `g` via `fabric_recv`** (`fabric_recv(g,[r]) @ weight`): FAILS to
  legalize `memref.reinterpret_cast`. (Old claim "fabric_recv composes only with
  eltwise, not matmul/copy_" is WRONG — see the 2026-06-26 re-investigation below.)

### Re-investigation (2026-06-26): the three blockers are narrower than thought

Probed each wall in isolation (`scratchpad/agf_bridge.py`, `agf_grid_probe.py`,
`agf_circ.py`, `agf_circ_storeonce.py`, `agf_circ_fixedloopstore.py`), compile-only
through d2m-to-ttkernel. The earlier "fabric_recv can't feed matmul" conclusion was
wrong. The real, separable facts:

1. **`fabric_recv` lowering IGNORES its indices** (LowerLoadStoreOpsToDMA
   `D2MLowerFabricRecvRewritePattern` = just `cb_reserve` + `cb_push`, no index).
   The `memref.reinterpret_cast` that fails is the **compute side** subviewing a
   block of a *multi-block grid operand* — it appears for ANY index kind (runtime
   `g[r]` AND grid `g[cy]`), because `g` has grid `[N,1]` (4 blocks). `remote_load`
   dodges it only by NoC-reading the block into a fresh local buffer.
   -> AVOIDABLE: receive into a **gridless `t = empty([MT,KT])` scratch** and
   `fabric_recv(t, [])` (no index, single block) — then `fabric_recv(t,[]) @ w`
   COMPILES. **Gridless fabric_recv DOES feed a matmul.**
2. **Fabric-sending a loop-carried value directly crashes** d2m-to-ttkernel (the
   `m3i` crash). AVOIDABLE the same way the ring does: forward a fresh **send-only
   copy** (`fwd = copy_(empty, cur)`), keep `cur` for compute only (also satisfies
   "a CB value feeds at most one DM op").
3. **An in-loop `remote_store` to an OUTPUT operand crashes** d2m-to-ttkernel —
   `agf_circ_storeonce.py` (store once AFTER the loop) COMPILES; moving that same
   store INSIDE the loop crashes, even at a FIXED index `out[0,0]`. This is the
   real, last blocker: the working ring stores its output exactly once after the
   loop; a fused AGMM must emit N output row-blocks (one per gathered row), which
   is inherently per-iteration / multi-block output.

So a single-core circulate-and-matmul fused kernel (`agf_circ.py`: compute-owned
`cur`, send-only `fwd` copy, `cur = copy_(cur, fabric_recv(t,[]))`, `cur @ w`)
gets all the way through EXCEPT the per-row output store (#3).

4. **HAND-UNROLLING to dodge #3 hits a fourth wall** (`agf_unroll.py`,
   `agf_unroll2.py`): with straight-line steps (no scf.for) the in-loop-store crash
   is gone, but a `copy_(c, fabric_recv(t,[]))` whose result `c` then feeds BOTH a
   matmul (`c @ w`) AND a forward-copy fails — "failed to legalize unresolved
   materialization from memref<...l1> to !ttkernel.cb". A fabric-received value
   bridged through `copy_` and fanned out to a matmul + a send doesn't resolve to a
   CB. This is the same family as the send-only-forwarding wall
   ([[d2m-ccl-send-only-forwarding]]) but now with a matmul consumer.

Net: blockers #1/#2 are solved at the DSL level; #3 and #4 are genuine compiler
gaps.

### Wall #3 FIXED (2026-06-26, commit 4a0140ad1)

#3 was NOT d2m-to-ttkernel — it was a null-deref CRASH in **d2m-allocate**
(`materializeAliasedLoadStore`, Allocate.cpp:1324, confirmed via a Debug build
backtrace): `isa<OperandAliasOp>(remoteStoreOp.getLocalBuffer().getDefiningOp())`
where the in-loop store's local buffer is a loop-carried scf.for iter_arg (a block
argument, `getDefiningOp()==null`). Fixed with `isa_and_nonnull`. Regression:
`test/d2m-jit/test_inloop_output_store.py` (lower-only). The minimal in-loop-store
repro now compiles; the circulate-matmul `agf_circ.py` now passes d2m-allocate.

### Wall #5 (the NEXT crash, after #3): d2m-to-ttkernel on matmul-of-fabric-recv

With #3 fixed, `agf_circ.py` (single-core circulate-MATMUL with in-loop store)
crashes one stage later, in `d2m-to-ttkernel-pre-emitc-pipeline` (pass #14,
OperationLegalizer::legalize) — the gridless `fabric_recv(t,[]) @ w` + in-loop
matmul lowering. This is the same area as wall #4 (the hand-unroll
`copy_(fabric_recv)`->matmul+forward materialization). So after #3, the remaining
fused-kernel blocker is the matmul consuming a fabric-received tile through the
d2m-to-ttkernel conversion. (The gather-only circulate WITHOUT matmul compiles end
to end now.)

Peeling #1→#2→#3 (and #3 now fixed) leaves the matmul-of-fabric-recv lowering as
the last gap for a single-core fused AGMM.

### Wall #5 FIXED (2026-06-26, commit ff918d716): mesh_position without an fcm

The pass-#14 crash was NOT matmul-specific. Debug-build backtrace:
`D2MToTTKernel.cpp:174 getFabricConnectionManager: Assertion 'fcm' failed` via
`D2MMeshPositionRewriter`. The fused kernel's output store index is
`(mesh_position(1) - 1 - k) % N`; after split-v2 that LOCAL store + its
mesh_position land on a different NoC thread than the fabric send, so its func had
no fabric op and D2MToTTKernelPass skipped fcm creation (it gated on `fabricOps`,
not `fcmUsers`). Fixed: create the fcm when a func has any fcm user (mesh_position
included) -- `setup_fabric_connections` opens only `num_send_dir` (=0) connections
for a non-sending thread, so it is just the topology build mesh_position needs.
Regression `test/d2m-jit/test_meshpos_local_store.py`.

### Walls remaining for the fused kernel (2026-06-26)

With #3 and #5 fixed, the fused AGMM gets further but two issues remain:

- **Wall #4 (compile, the last blocker) — DEEP-DIAGNOSED 2026-06-26**: a matmul
  that reads a fabric-received value fails d2m-to-ttkernel: "unresolved
  materialization memref<1x2xtile,l1> -> !ttkernel.cb remained live", live user =
  `ttkernel.copy_tile` (the matmul operand unpack). Isolation (`scratchpad/w4_iso.py`
  MODE A/B/C): the trigger is the **matmul reading a fabric-recv'd operand itself** --
  MODE A (`fabric_recv(t,[]) @ w`, no copy_, no fanout) fails identically to the
  copy_-bridge and fanout variants. ROOT CAUSE (debug-build backtrace + unified-IR
  inspection): the matmul's activation operand is a `remote_store`-DST scratch
  (the ring send/recv buffer `t`; `fabric_recv` folds to reading `t` directly before
  split-v2). Because `t` is a fabric-write target it is threaded as a GENERIC OPERAND
  (`d2m.get_arg` -> a buffer ADDRESS), not a hoisted CB (`d2m.get_cb`). The weight
  works because it comes from `remote_load` -> a proper hoisted CB read via `d2m.wait`.
  d2m-to-ttkernel's matmul `getCB` then casts the operand memref to a CB, leaving the
  unresolved cast. Eltwise (the ring) works because it reads the scratch via
  `memref.load`->`copy_tile`, a different CB-resolution path than the matmul's
  `getCB`/`tile_matmul_block` operand handling. FIX is NOT localized: a matmul
  consuming a fabric-recv'd / `remote_store`-DST operand needs that operand
  represented as a real CB (hoisted, with a wait) rather than a bare get_arg buffer
  address -- a split-v2/HoistCBAllocs + d2m-to-ttkernel matmul-operand change, with
  regression risk to the core matmul path. (A first attempt -- tracing through
  `FabricRecvOp` in split-v2's `traceComputeMemrefToCB` -- was moot: `fabric_recv`
  folds away before split-v2 runs.)
- **Runtime hang (scf.for variant only)**: an in-loop `remote_store` to a grid
  output compiles now (wall #3 fixed) but HANGS on device (the output CB handshake
  per iteration). The HAND-UNROLLED variant (straight-line stores) avoids this, so
  it is the better target -- its only remaining blocker is wall #4.

So the path to a working single-core fused AGMM is: fix wall #4, then the
hand-unrolled `agf_unroll2.py` should compile and run (it sidesteps the runtime
hang). Non-fused two-kernel port remains the shipped deliverable.

### UPDATE 2026-06-30: WALL #4 IS FIXED on this branch (split-threads rebase)

Re-verified with a fresh minimal repro `test/d2m-jit/_w4_iso.py`: a matmul that
DIRECTLY consumes a fabric-received operand -- `g = fabric_recv(tmp,[]); g @ w` --
now **compiles, runs, and is correct (PCC 1.0 on all 4 devices of the 1x4 ring).**
So the single-consumer fabric_recv->matmul primitive (the core of a fused
overlapped AGMM) WORKS. (Wall #4 was the documented last blocker; the
split-threads rebase / R-fixes resolved it.)

The fused overlapped ring AGMM (`test/d2m-jit/_agmm_fused.py`: each device matmuls
its current shard while the fabric forwards/receives the next) now has TWO narrower
remaining gaps, both reproduced minimally:

1. **FANOUT wall** (`_w4_iso.py` fanout mode; the hand-unrolled fused kernel hits
   this). When the fabric-received value feeds BOTH a matmul AND a forward-send
   (the ring-circulate: receive shard, matmul it, re-send it onward), compile fails
   in **d2m-allocate** `analyzeGenericRegionAllocs` (`Allocate.cpp:662`): "Alloc op
   not tagged with any recognized attributes" -- an in-generic `memref.alloc` (from
   the fanout) reaches Allocate untagged (neither `d2m.scratch_buffer` nor
   `d2m.synchronized_buffer`). An earlier split/fusion pass (MarkSynchronizedBuffers
   / split-v2 / scratch marking) fails to tag it because the recv value fans out to
   two consumers. Single-consumer (matmul only, OR eltwise+forward as in the ring
   all_reduce) is fine -- it's specifically matmul-consumer + forward fanout.
2. **Looped in-place circulate races** (`_agmm_fused.py` default, runtime scf.for).
   It COMPILES (loop-carried `cur` updated in place via `copy_`, like the ring
   all_reduce) but is WRONG on device (PCC ~0.44, all rows ~0.5) -- a compute<->DM
   ordering / in-place-`cur`-reuse race (the forward-send reads `cur` while the next
   iter's copy overwrites it; the matmul reads mid-update). The hand-UNROLLED
   variant (distinct SSA buffers per step, no reuse) would avoid this race but hits
   the FANOUT wall (#1) at compile.

So: fixing the FANOUT alloc-tagging (#1) should let the hand-unrolled fused ring
(distinct buffers) compile AND run correctly -- the clearest path to a real fused
overlapped AGMM. Repros: `_w4_iso.py` (wall #4 fixed + fanout), `_agmm_fused.py`
(looped + `unroll` mode).

### UPDATE 2026-06-30 (cont.): fused ring now COMPILES; device hang is the blocker

The FANOUT compile wall was an artifact of the isolation repro (its forward send dst
was never `fabric_recv`'d, so untagged). In a REAL ring the send dst IS the next
recv source, so it gets tagged. Reworking the hand-unrolled fused ring
(`_agmm_fused.py` `unroll` mode) to matmul the `fabric_recv` value DIRECTLY (the
`_w4_iso` form, `g @ w`, NOT a `copy_(empty, fabric_recv)` bridge) and forward a
fresh `copy_` of it -> **it now COMPILES through the whole pipeline** (past the
prior session's materialization wall). 

But it **HANGS on device** (4 devices, compile + fabric-init OK, then no completion).
The hang is the multi-step ring pattern: 3 fabric forwards + 3 recvs + 4 matmuls + 4
output stores at dynamic indices `(p-k)%N`. Key suspect: re-forwarding a
FABRIC-RECEIVED value (`g1 -> copy_ -> send`), which the working ring all_reduce
deliberately AVOIDS (it forwards a fresh COMPUTE output `acc-acc_prev`, never the raw
recv). Same hazard family as the ring readback hang
([[d2m-ring-interleaved-fabric-hang]]) and the chunked-AG hang: fabric DM <-> local
store/compute NoC contention + per-step ordering. `_w4_iso` (1 send, 1 recv, 1
matmul, 1 store, NO re-forward) runs fine; the multi-step re-forward + multi-store
hangs.

DEVICE HANG ROOT-CAUSED (2026-06-30) by step bisection of the straight-line unrolled
ring (`_agmm_fused.py unroll steps=K`): **steps=1 COMPLETES, steps=2 HANGS.** Step 2
is the first time a fabric-RECEIVED value `g1` FANS OUT to both a matmul (`g1 @ w`)
AND a forward-copy (re-send). So the hang = **fanout of a fabric_recv'd value to a
matmul + a re-forward.** The ring all_reduce never hangs because its received value
has a SINGLE consumer (`acc += r`); the forward is a separate compute output. Ring
all-gather inherently re-forwards each received shard -> inherent fanout -> hang.
(steps=1 forwards only the OWN shard, a remote_load, not a recv -> fine.)

### Full fused-AGMM constraint map (2026-06-30) -- every DSL path hits a distinct wall

Tried all three structures at [1,1,1]:
1. **Ring circulate** (`_agmm_fused.py`): re-forwards each received shard -> FANOUT
   of a fabric_recv value to matmul+forward -> **device HANG** (bisected above).
   Looped (scf.for) variant: in-place `cur` reuse -> compiles but races (PCC ~0.44).
2. **Mcast + `remote_load`** (`_agmm_mcast.py`): each device mcasts its own shard to
   all peers (one `device_mcast_shape=[1,N]`, no worker re-forward -> NO fanout, NO
   hang!), matmuls own (overlaps gather), waits, then `remote_load(g[q]) @ w`. RUNS
   but **WRONG (PCC ~0)**: the in-program `remote_load(g)` reads a DIFFERENT buffer
   than the DM thread's fabric-mcast target (split-v2 doesn't alias them) + no
   compute<->DM ordering -- the documented within-one-program gather->matmul issue.
3. **Mcast + `fabric_recv`** (the correct in-kernel consume): `fabric_recv(g,[q]) @ w`
   of a grid slot -> **WALL #1** still present: `memref.reinterpret_cast` fails to
   legalize in d2m-to-ttkernel (compute subviewing a multi-block grid operand).
   Gridless single-block fabric_recv works (`_w4_iso`), but the mcast writes N grid
   slots and they can't be addressed as N distinct gridless scratches (the send
   target is the sender's mesh_position p, can't index N distinct `empty()` allocs).

So a correct FUSED overlapped AGMM needs ONE compiler fix. RECOMMENDED: fix **wall #1**
(legalize `fabric_recv` of a multi-block grid slot, i.e. the compute-side
`reinterpret_cast` subview) -> unblocks path #2/#3 (mcast gather + per-slot
fabric_recv matmul: no fanout, correct, own-shard matmul overlaps the gather). The
mcast structure (`_agmm_mcast.py`) is otherwise complete and hang-free. Alternative:
a within-program fabric-write->compute-read barrier + buffer aliasing so path #2's
`remote_load(g)` is ordered and reads the fabric-written buffer (heavier, like the
[[d2m-ring-interleaved-fabric-hang]] InsertSpillAndScratch barrier).

Repros: `_w4_iso.py` (fabric_recv->matmul OK; fanout), `_agmm_fused.py`
(ring, `unroll steps=K`), `_agmm_mcast.py` (mcast gather, remote_load vs fabric_recv).

### Wall #1 DEFINITIVELY root-caused (2026-06-30): grid-operand vs CB representation gap

Dug all the way down on the mcast path (`_agmm_mcast.py`). Findings, in order:
- The matmul-read path #2 (`remote_load(g[q]) @ w`): the gathered buffer `g` (kernel
  operand, grid [N,1]) is NOT split into two buffers -- post-bufferize the mcast
  `remote_store` writes get_arg(2) and the matmul's `dma_read` reads get_arg(2), the
  SAME buffer, ordered by `semaphore_wait`. Yet **the GATHER ITSELF is wrong**:
  isolating it (copy `g[q] -> out[q]`, NO matmul) gives PCC ~0 on all 4 devices. So
  within ONE generic, a `remote_load`/`dma_read` (noc read) of `g` does NOT see the
  fabric mcast write -- the noc read races/conflicts with the fabric connection on
  the DM thread (the [[d2m-ring-interleaved-fabric-hang]] read-back hazard). The
  non-fused 2-kernel port is correct ONLY because the AG kernel completes (program
  boundary) before the MM kernel's read. So path #2 is fundamentally wrong for a
  fused kernel, regardless of buffers/ordering.
- `fabric_recv` is the DESIGNED correct in-kernel consume (view-free, exposes the
  fabric-written buffer with NO noc read -- proven by `_w4_iso`). But it only works
  for a GRIDLESS single-block scratch: an in-kernel `empty()` becomes a hoisted CB,
  `fabric_recv` = cb_reserve+push, matmul reads the CB. The mcast gather needs a
  GRID [N,1] buffer to address slots `[p,0]` -> that is a kernel OPERAND -> lowers to
  `d2m.get_arg` (a buffer ADDRESS), NOT a CB. The matmul LLK requires a CB, so the
  per-slot read becomes a `memref.reinterpret_cast` of the get_arg grid operand,
  which d2m-to-ttkernel can't legalize (wall #1). `getCB` (D2MToTTKernel.cpp:235)
  has no reinterpret_cast case; `traceComputeMemrefToCB` (SplitUnifiedThreadV2.cpp:62)
  traces through reinterpret_cast but stops at `operand_alias` and finds a get_arg
  grid operand, not a CB.

ROOT GAP: fabric-gathered data lives in a MULTI-BLOCK GRID OPERAND (get_arg,
fabric-writable, slot-addressable by the mcast), but the matmul needs a SINGLE-BLOCK
CB, and the only bridge that works in-kernel (`fabric_recv`) requires a gridless CB.
There is no mechanism to expose a grid-operand slot as a matmul CB. In-kernel
`empty()` scratch -> CB but is gridless (can't be mcast-slot-addressed); kernel grid
operand -> slot-addressable but get_arg (not a CB).

So wall #1 is NOT a localized patch -- it is a representation change. Two options:
  (A) make `fabric_recv` of a grid-operand slot expose that slot as a CB to the
      matmul (split-v2 / HoistCBAllocs: back the grid operand with per-slot CBs or a
      multi-slot CB + index the matmul read; high regression risk to the core matmul
      operand path, as the prior session warned).
  (B) make a within-generic `remote_load(g)` (noc read) reliably see the fabric mcast
      write (a flush/ordering fix) -- but this fights the established direction
      (the read-back-vs-fabric hazard is exactly why `fabric_recv` exists).
RECOMMENDATION: (A). It is the principled fix and matches how the TTNN op routes
gathered data into the matmul's input CB. It is a multi-pass change, not a one-liner.

### BREAKTHROUGH 2026-06-30: dedicated-fabric-worker fused AGMM WORKS

The dedicated-fabric-worker (router_cores) path -- already implemented on this branch
(attr + `is_router_core()`/`router_direction()` intrinsics + flatbuffer + runtime
subset + lowering fcm gating) -- gives a CORRECT FUSED single-generic all_gather +
matmul. `test/d2m-jit/test_all_gather_matmul_fused.py` (repro `_agmm_router.py`),
6 shapes PASS on the full 1x4 ring (M=128, K up to 256, Nout up to 1024).

Structure (the fix for the gather-within-one-generic hazard): split the two roles
across cores so neither conflicts.
  - `is_router_core()` gates the fabric all_gather to ONE core (0,0): it mcasts this
    device's shard to slot p (`remote_store(g,[p,0], device_mcast_shape=[1,N])`),
    which lands in CORE p's L1 -- so the gathered tensor `g[N,1]` is DISTRIBUTED
    across cores 0..N-1, not resident on the router.
  - every COMPUTE core reads its OWN local slot `g[cy]` and matmuls it. Compute cores
    NEVER hold the fabric connection, so their noc read of the gathered data SEES the
    fabric write (this is exactly what `_agmm_mcast.py` got wrong: there the single
    (1,1)-grid core did both the fabric mcast and the read -> read-back-vs-fabric
    hazard -> gather garbage. Separating the roles across cores fixes it.)
  - a `ready` global semaphore fences gather->matmul: the router
    `semaphore_set(ready, 1, core=[0,0], mcast=[N,1])` after the gather; all cores
    `semaphore_wait(ready, 1)`. (NOTE: `semaphore_inc` CANNOT multicast --
    D2MToTTKernel asserts "semaphore_inc multicast is illegal"; only `semaphore_set`
    mcasts. Local inc and single-remote set are also illegal -- pick set for
    mcast/local, inc for single-remote.)
  - fabric: `routing="unidir_ring_torus"` needs cores_per_link=2 ->
    `router_cores=[(0,0),(1,0)]` (fwd+bwd slots).

This is FUSED (one generic, no AG/MM program boundary) and the structure that
enables overlap.

### SCALING the fused kernel (2026-06-30): 2D grid + distributed weight

`_agmm_router2d.py` (committed in `test_all_gather_matmul_fused.py`): grid (N, gn)
instead of (N,1). The TP weight is DISTRIBUTED across the gn columns (w grid [1,gn],
each core holds only w[:, cx-block] = [KT, NTd/gn]) so no core resident-holds the
whole weight -- the per-core weight L1 that capped the (N,1) kernel. Gather unchanged
(router (0,0) mcasts shard p to g[p]); compute core (cy,cx) reads its gathered row
g[cy] (NoC read from core (cy,0)) and weight block w[0,cx], matmuls -> out[cy,cx].
Scales to **K=1024 (KT=32), Nout=4096 (NTd=32), M=256 (MT=2)** on the 1x4 ring
(vs K=256/Nout=1024 for the (N,1) form). Next limit = the per-core single-block
matmul over full K (gather g[cy]=[MT,KT] resident); KT>=40 / NTd=128 overflow.
Lifting it needs K-accumulation via a K-chunked gather (g grid [N, nK], router writes
g[p,:] in chunks) -- the chunked-AG machinery merged in.

### OVERLAP attempts (2026-06-30): blocked; needs the ring-gather redesign

Current fused form fences the full gather before the matmuls (coarse; only the
weight loads overlap the gather). Two attempts at fine-grained overlap, both blocked:
- **Early own-shard write** (`_agmm_overlap.py` v1): write own s_p to g[p] locally +
  signal row p early, before the fabric mcast. BLOCKED: `core_write` verifier --
  "dst must be a local buffer (no device layout)"; g is a device-layout operand.
- **Own row from in0** (`_agmm_overlap.py` v2): out[p] = in0 @ w (own shard is local,
  no gather) on a runtime `if cy == p` branch, overlapping the peer gather. BLOCKED:
  trisc0 build failure -- the divergent branch puts a DM op (`remote_load(in0)`) on
  the compute path, which the thread-split can't lower ("NOC_INDEX not declared").
The clean overlap needs a RING gather: the router receives shards one-per-step
(fabric) and signals each as it lands, compute cores matmul each -- the router does
ALL fabric (receive + re-forward), compute cores only matmul, so no fanout-matmul
hang and real pipelined overlap (router does step t+1's fabric while compute matmuls
step t's shard).

### Ring-gather overlap ATTEMPTED (2026-06-30): 3 distinct primitive walls

`_agmm_ring.py` (hand-unrolled N=4 ring on the router; router fabric-recvs each shard
+ re-forwards; per-row `ready` signal; compute core cy matmuls its row when ready).
The ring control flow + fabric all work; the blocker is DELIVERING each received
shard to its compute core's matmul-input, hitting a different wall each way:
1. **`core_write` to an in-kernel `gb` scratch, matmul reads `gb`** -> COMPILES but
   HANGS on device: `core_write` writes raw L1 but does NOT push a CB, so the
   compute matmul's `cb_wait_front(gb)` never fires (the matmul needs its input via a
   CB-pushing op -- remote_load/fabric_recv -- not a raw core_write).
2. **Local `remote_store(g,[q,0], shard)` to the device-layout g** -> compile error
   "remote_store memref must be remote (have a device layout)": a LOCAL (no-device-
   args) remote_store may only target the OUTPUT operand; g is an input.
3. **Cross-device-to-self `remote_store(g,[q,0], ..., device_mcast_shape=[1,1],
   start_device=[dy,p], semaphore=ready)`** (reuses the working remote_load g path,
   ready inc ordered after the write) -> runtime TT_FATAL program.cpp:2483
   "state.offset <= max_size": the N self-targeted mcast stores each spin up fabric
   config -> fabric runtime-arg overflow (a ring inherently needs N fabric ops vs the
   mcast's 1, and self-mcast is the wrong tool for a local slot write).
Net: overlap needs a primitive that delivers a router-gathered shard into a compute
core's matmul-input CB (a CB-pushing core->core handoff, or a local g-slot write that
isn't modelled as fabric). The non-overlapped fused kernel + the 2D distributed-weight
scaling are the shipped deliverables; fine-grained overlap is gated on that primitive.

### The CB-pushing handoff ALREADY EXISTS (2026-06-30) -- verified working

It turns out d2m's `core_read`/`core_write` already provide the CB-pushing core->core
handoff; no NEW op is needed. Verified with two single-device probes (PCC 1.0):
- `_coreread_mm.py`: core 1 `core_read`s core 0's buffer and MATMULS it. Works --
  split-v2 wraps core_read as reserve+core_read+PUSH (SplitUnifiedThreadV2.cpp:250),
  so the matmul's input CB is pushed and consumed. (Caveat: `core_read` addresses by
  CORE coord, NOT by slot/offset -- it reads a whole buffer from a core; shards must
  therefore live one-per-core, not as slots of one buffer.)
- `_corewrite_mm.py`: core 0 `core_write`s a tile into core 1's LOCAL `gb` (in-kernel
  empty, non-device-layout -> core_write legal) + incs a ready sem; core 1 waits, then
  self-`core_read`s its own `gb` into a CB (which pushes it) and MATMULS. Works.
So the push handoff = router `core_write`s shard -> peer's local `gb`; peer
self-`core_read`s `gb` -> matmul. Both halves work standalone.

### Ring overlap with the handoff: hang isolated to the fabric re-forward (2026-06-30)

`_agmm_ring.py` wires the handoff into the ring: router fabric-recvs each shard,
`core_write`s it to that row's core `gb`, incs `ready[q]`; compute core self-core_reads
`gb` + matmuls. COMPILES, but HANGS on device. Since both handoff halves work
standalone (probes above), the hang is in the RING FABRIC part -- specifically the
re-forward of a fabric-RECEIVED value (`g1 -> core_write(gb) + copy_(forward)` fanout),
the same fanout-of-fabric-recv hazard as the original wall (a fabric_recv'd value
feeding two consumers). The ring inherently re-forwards each received shard while also
delivering it locally -> inherent fanout. RESOLUTION PATHS: (i) break the fanout (e.g.
forward a value re-read from the recv scratch via a separate op, or double-buffer so
the forward and the core_write read distinct buffers); (ii) avoid re-forward entirely
with the device_mcast gather (each device mcasts its own shard to all -- no fanout, but
all shards arrive together so less ordered overlap). The handoff primitive is DONE; the
overlap is gated on the ring's re-forward-fanout device hang.

### Path (i) break-the-fanout + path (ii) no-re-forward BOTH attempted (2026-06-30): still hang

- **(i) Copy the recv to a COMPUTE value before fanning out** (`_agmm_ring.py`:
  `c1 = copy_(empty, fabric_recv(t1)); core_write(c1, gb); f2 = copy_(empty, c1)`):
  still HANGS. So it is not specifically a raw-fabric-recv fanout -- the hang is the
  ring's combination of fabric re-forward + cross-core `core_write` (to a peer's gb)
  + per-slot `semaphore_inc` to a remote core, ALL while the router core holds the
  fabric connection (a NoC-vs-fabric-connection contention on the router, the same
  hazard family as the read-back hang).
- **(ii) No re-forward via per-slot mcast signalling** (`_agmm_overlap2.py`: each
  device mcasts its own shard once, `semaphore_indices=[p,0]` to signal core p):
  HANGS. Root cause: the mcast's completion semaphore increments the SENDER's core on
  receivers, NOT an arbitrary target slot -- so only core 0 (the router's core) gets
  incremented N times; cores 1..N-1 never get their per-slot signal and wait forever.
  The fabric gives a per-arrival COUNT on the sender's core, not which-slot-landed, so
  receive-side per-slot signalling is not available from the mcast.

NET (overlap, honest): the CB-pushing handoff works (verified), the non-overlapped
fused + 2D scaling ship, but every overlap orchestration hangs on device:
  - ring (router knows arrival order -> can signal per-slot) hangs on the router's
    fabric + cross-core core_write/sem-inc contention;
  - mcast (no re-forward, no contention) can't do per-slot receive signalling.
Resolving it needs DPRINT-level device debugging of the router's NoC/fabric/semaphore
interplay (which exact op stalls) -- a focused follow-up, not a kernel-DSL fix. The
shipped deliverables (committed tests, 14 green) stand; overlap remains open with the
two dead-ends above precisely characterized.

### DPRINT instrumentation + ring stall LOCALIZED (2026-06-30)

Added a kernel-side marker print: `dprint("msg\\n")` (api.py, `@syntax("dprint")` ->
`d2m.print_` -> ttkernel.dprint; NOTE double-escape newlines as `\\n` or the emitted
C++ gets a literal newline -> trisc compile error). Enable:
`TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_FILE=<path>`. Verified working (markers
print). Reusable for any kernel hang.

Instrumented the ring (`_agmm_ring.py`) with per-step markers and bisected the hang.
FINDING: every router reaches `R snd2` (issues the step-2 fabric forward) but
`R got2` NEVER fires -- all routers stall at the step-2 `semaphore_wait` (the 2nd
forward's arrival ack). `R got1`/`snd1` fire (step 1 completes). So the stall is the
**2nd fabric send's completion** on the router. Systematically RULED OUT (each
re-tested, still stalls at got2):
  - the cross-core deliveries (moved core_write+ready-inc to AFTER the ring);
  - dual vs single router core (router_cores=[(0,0),(1,0)] vs [(0,0)]);
  - cumulative `es` (wait 1,2,3) vs separate per-step semaphores (es,es2,es3 each ==1);
  - forward via `copy_` vs eltwise `+ zeros` (the _m3b/all_reduce send-only pattern).
The ONE remaining difference from the WORKING multi-send ring (test_ring_all_reduce_loop,
3 sends, grid (1,1), NO router_cores): that runs whole-grid fabric on a single core;
mine runs `router_cores`-gated fabric on a `(N,1)` grid. PRIME SUSPECT: the
router-core fabric setup (fcm gated to the router subset, fabric_config.cpp
`appendFabricConfigArgs` over router_cores) does not support MULTIPLE sequential
fabric sends from the router the way the whole-grid (1,1) setup does -- the 2nd
send's connection/ack stalls. NEXT: confirm by running the bare ring fabric (3
forwards+recvs, no matmul/gather) on `(N,1)`+router_cores vs grid (1,1); if (1,1)
multi-send works and router-gated does not, the fix is in the router-core fabric
connection setup (runtime/lib/common/fabric_config.cpp + the fcm lifecycle), not the
kernel. The `dprint` primitive + this bisection are the tools for that follow-up.

So the two workstreams have merged (fused router kernel + distributed-weight
scaling); remaining: K-accumulation gather (full K) and the ring-gather overlap.

IR INSIGHT (`scratchpad/agfdump2.py`): a single `@d2m.kernel` with AG +
`static_range(N)` matmuls does NOT become one generic -- it DECOMPOSES into the AG
generic + N matmul generics (one per UNROLLED iter; this was the trace-time unroll,
now removed -- a runtime `range(N)` would instead be ONE matmul generic with an
scf.for, which is the retry path noted above), with `g` flowing between them
*within one program*. That is exactly why the non-fused two-kernel version is
correct (the AG kernel hits a program boundary / device_synchronize before the MM
kernel) while this is not (no ordering between the AG generic's fabric writes and
the matmul generics' reads of `g` inside one program). So "fuse into one kernel"
splits into two sub-problems: (i) one TRUE fused generic (DM gathers, compute
matmuls via fabric_recv) -- blocked by the fabric_recv->matmul legalization wall
above; or (ii) keep AG-generic -> matmul-generic but add a cross-generic
fabric-write->compute-read barrier within the program (analogous to the
InsertSpillAndScratch fusion barrier in [[d2m-ring-interleaved-fabric-hang]]).

CONCLUSION (updated 2026-06-26): the circulate-and-matmul fused kernel
(`scratchpad/agf_circ.py`) is correct-by-construction and compiles through every
stage EXCEPT the per-row output store. The ONE remaining framework gap is:

  **(C) d2m-to-ttkernel must support an in-loop / per-iteration `remote_store` to
  an output operand** (today only a single post-loop store works, as in the ring).

With (C), the single-core fused AGMM works for small shapes immediately. Two
alternative framework fixes also unblock fusion but are heavier:
  (A) handle the multi-block grid-operand `reinterpret_cast` so compute can consume
      a grid `g[cy]` via `fabric_recv` (enables a grid=(N,1) fused form), or
  (B) a cross-generic fabric-write->compute-read barrier so the natural
      AG-generic -> MM-generic `remote_load(g)` is ordered (+ g aliases) — the
      remote_load path otherwise races (rel 1.37).
This mirrors how the TTNN op solves it: dedicated fabric workers gather into
compute-readable buffers with explicit cross-core sync. The WORKING port stays the
non-fused two-kernel `test_all_gather_matmul.py`; the fused single-kernel is gated
on fix (C) (smallest) or (A)/(B).

## Designs for the fused single-kernel AGMM (after the mesh matmul works)

Two reconciliation points for AG (writes `[1,K]` rows on an `[N,1]` grid) vs MM:
1. **Row-parallel MM** on the `[N,1]` grid (no reblock): `grid=(N,1)`, core cy
   does `gathered[cy] @ weight -> out[cy]`. Natural fit for the gather layout.
2. **Reblock** the `[N,1]` gathered to one `[M,K]` block then a normal matmul
   (hit a buffer mismatch via chaining; revisit).
Fusing AG+MM in ONE kernel additionally needs compute to read the
fabric-gathered buffer (fabric_recv-style) and, for a grid, the gathered data
available to each matmul core (cores don't share L1) -- this is what the TTNN op
solves with dedicated fabric workers + careful routing.

## Hardware note

The every-core-does-fabric ring all_reduce already converged at <=4 fabric cores
(2 eth channels between adjacent devices, num_links<=2; see
d2m-ccl-send-only-forwarding memory + test_all_reduce_grid.py). Scaling AGMM to
the reference's large shapes ultimately needs the dedicated-fabric-worker CCL
design, not every-core-fabric.

## Scaling the non-fused port (2026-06-30): distributed tiled matmul + the AG wall

The committed `test_all_gather_matmul.py` was upgraded from the single-block matmul
(which capped at K=512 / Nout=512 -- a single `[MT,KT]@[KT,NTd]` block plus the
single-core TP weight overflow L1) to a **distributed tiled matmul**, validated on
this 1x4 Blackhole:

- **Distributed tiled matmul** (`_make_mm`): matmul grid `(gm,gn)`; core `(cy,cx)`
  accumulates `acc[mb,nb] += g[cy,k] @ w[k,cx]` over `nK` K-sub-blocks via NoC
  reads. Only `[mb,KB]+[KB,nb]` tiles are transient per step, so L1 stays small for
  large K. `gm<=10`, `gn<=8`, and every operand grid (`g:[gm,nK]`, `w:[nK,gn]`,
  `o:[gm,gn]`) must be `<=110` cores (the BH worker grid is 10x11). `mb/nb` absorb
  M/N beyond the grid.
- **Distributed weight**: the TP weight slice is sharded across the `[nK,gn]` grid
  AT `mesh_shard` time (block `[KB,nb]`). A single-core weight (`grid [1,1]`)
  residents the whole `[K, Nout/N]` slice on one core and overflows L1 as K grows
  (the first wall after the matmul fix: ~2MB at K=1024).

Validated single-axis envelope (PASS, PCC ok): **K -> 2560** (KT=80), **Nout ->
16384** (NTd=128), **M -> 1024** (MT=8). N scales essentially freely because the
all_gather never touches N (the gathered tensor is `[M,K]`; the weight is
TP-sharded and only read by the matmul).

THE REMAINING WALL = the **all_gather residency**. The AG places each device's full
`[MT,KT]` activation shard on a SINGLE fabric core (`grid (1,1)`, one mcast). So
`MT*KT` tiles must fit that core's L1 (minus the tilize-staging buffer +
matmul) -- roughly `MT*KT <~ 150-200` tiles. Hence K=5120 (KT=160, 160 tiles + 640KB
staging ~ 2MB) and M=2048 (MT=16 -> 128-tile rows) and any combined `MT*KT>~200`
(e.g. (8,32,16) = 256 gathered tiles = 4.2MB) FAIL at L1 allocation.

Lowering AG residency needs spreading the gathered shard across more fabric cores,
but the box has only **2 fabric routing planes** (`num_links=1` -> `cores.size() <=
num_links * cores_per_link = 2`), so >2 fabric cores is rejected at runtime
("Number of cores to connect to fabric routers exceeds number of routing planes").

A **chunked-loop AG** (single fabric core looping `for kc in range(nChunks)` over K
or M chunks, residency = one chunk, cumulative semaphore) was tried: it COMPILES and
is correct for small chunk counts (KT=16/40 passed) but **HANGS intermittently on
device** for larger chunks. Repro: `test/d2m-jit/_ag_chunk.py` (AG-only, no matmul;
`python _ag_chunk.py [sync] KT KB`).

### Chunked AG: the hang IS fixable (per-chunk barrier), but the true limit is fabric resources, not L1 (2026-06-30)

Isolated in `_ag_chunk.py`. The intermittent hang is the looped AG issuing many
fabric mcasts with only ONE final `semaphore_wait((N-1)*nChunks)` -- too many large
mcasts in flight saturate the fabric and some never land, so the count never
completes. FIX: a per-chunk CUMULATIVE barrier inside the loop --
`semaphore_wait(es, (kc+1)*(N-1))` after each chunk's `remote_store` (bounds
outstanding mcasts to one chunk). NOTE the variant must be selected in PYTHON at the
factory level (`if sync: @kernel def ag...`); a traced `if` over a bool becomes a
runtime `scf.if` and fails ("Cannot compare non-integer values").

With the per-chunk barrier the reliable envelope is ~**<=8 chunks AND <=10 tiles
(40KB f32) per chunk**:
  - 8 chunks x 10 tiles (KT=80): PASS (deterministic). 8 x 12 (KT=96): HANG.
  - 10 chunks x {8,10} tiles (KT=80/100): HANG (regardless of payload).
  - 2 chunks x 40 tiles (KT=80, no sync even): PASS (a single big mcast is fine).
So BOTH a per-mcast payload ceiling (~10 tiles) AND a mcast-count ceiling (~8 with the
barrier) bind -- the signature of a fabric in-flight / buffer-credit limit
(`num_links=1`, a single fabric core, `noc0`), NOT L1 residency.

CONSEQUENCE: chunking trades L1 residency for more/smaller fabric transfers, but on
THIS box the fabric resource limit binds at ~80 tiles total either way, so the
reliable chunked AG does NOT extend K beyond the non-looped AG's ~K=2560:
  - non-looped AG: ONE 80-tile mcast works; KT=160 (one 160-tile mcast) is fine for
    fabric but overflows L1 (~2MB residency).
  - chunked AG: avoids the L1 overflow but K=5120 needs 16 chunks of 10 tiles (>8
    chunk ceiling) or 8-11 chunks of >=15 tiles (>10 payload ceiling) -> hang; and
    >11 chunks also exceed the 11-wide worker grid (nChunks = gathered-tensor grid
    columns).
The fabric ceiling is the real wall. `num_links>1` would raise it but needs a
multi-core fabric grid with `router_cores` slot assignment (`fabric_config(...,
num_links=2, router_cores=[...])`) -- i.e. the dedicated-fabric-worker design, not a
single `(1,1)` AG core.

Sweep tool: `test/d2m-jit/_agmm_scale.py` (`python _agmm_scale.py K|N|M [filter]`);
AG-only hang repro: `test/d2m-jit/_ag_chunk.py`.

## Next steps

1. **Break the fabric ceiling** to reach K=5120 / M=115200: the dedicated-fabric-
   worker CCL design + `num_links>1` (a few fabric worker cores with `router_cores`
   slot assignment stream the gather into compute-readable buffers; matmul cores
   consume them -- the TTNN op's approach, `num_buffers_per_channel=48`,
   `chunks_per_sync=16`). The per-chunk barrier in `_ag_chunk.py` is the in-loop
   ordering primitive that design will reuse. (The DSL-level chunked AG is correct
   but the single-core/num_links=1 fabric caps it at ~80 tiles.)
2. Add bias / gelu / addcmul (the reference's fused epilogues) to `_make_mm`.
3. Fuse AG+MM into one kernel (blocked on wall #4, above).
4. Async overlap; then a 2D-mesh SP x TP split (needs a 2D mesh box).
