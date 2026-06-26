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

- **Wall #4 (compile, the last blocker)**: `c = copy_(c, fabric_recv(t,[]))` whose
  `c` is then consumed by a matmul AND a forward-copy fails d2m-to-ttkernel:
  "failed to legalize unresolved materialization from memref<1x2xtile,l1> to
  !ttkernel.cb that remained live" (both the hand-unrolled `agf_unroll2.py` and the
  scf.for `agf_circ.py`). This is the send-only-forwarding materialization wall
  ([[d2m-ccl-send-only-forwarding]]) extended to a matmul consumer -- a deeper
  d2m-to-ttkernel/split-v2 conversion change, not a localized null/gate fix.
- **Runtime hang (scf.for variant only)**: an in-loop `remote_store` to a grid
  output compiles now (wall #3 fixed) but HANGS on device (the output CB handshake
  per iteration). The HAND-UNROLLED variant (straight-line stores) avoids this, so
  it is the better target -- its only remaining blocker is wall #4.

So the path to a working single-core fused AGMM is: fix wall #4, then the
hand-unrolled `agf_unroll2.py` should compile and run (it sidesteps the runtime
hang). Non-fused two-kernel port remains the shipped deliverable.

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

## Next steps

1. Fix the multi-device-mesh matmul buffer-descriptor mismatch (mmonly.py) ->
   non-fused AG+MM end to end on small shapes.
2. Scale M/K/N; add bias, gelu.
3. Fuse AG+MM into one kernel (row-parallel design #1).
4. Add async overlap; then a 2D-mesh TP weight split (needs a 2D mesh box).
