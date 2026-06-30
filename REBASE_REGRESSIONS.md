# Rebase regression triage — nsmith/d2m-ccl2

Rebased onto new base `18908b135` (`origin/nsmith/split-threads` — "Decompose
SplitUnifiedThread into 3 passes; remove synchronized_region"). Old base was
`9885724e9`. Pre-rebase tip (ORIG_HEAD) = `9533de15`.

## R1 — d2m-jit pipeline lost `use-split-unified-thread-v2=1` and `use-tensor-accessor-dma=1` (HIGH)

**Status:** confirmed code regression; runtime impact pending test run.

`tools/d2m-jit/_src/builder.py` and `_src/config.py`: the rebase resolved the
conflict in favor of the **new base's** versions, which dropped:

- `config.use_split_unified_thread_v2` (default **True**) and its injection of
  `use-split-unified-thread-v2=1` into `d2m-be-pipeline`.
- `config.use_tensor_accessor_dma` (default **True**) and its injection of
  `use-tensor-accessor-dma=1` into `d2m-be-pipeline` and
  `d2m-to-ttkernel-pre-emitc-pipeline`.

Pre-rebase `_build_pipeline()` built `be_opts` dynamically; post-rebase
`_pipeline_passes()` hardcodes `d2m-be-pipeline{use-tile-matmul=...}` only.

Pass-level default of `use-split-unified-thread-v2` is **false**
(`D2MPipelines.h:239`), so the d2m-jit pipeline now runs the legacy/umbrella
`createD2MSplitUnifiedThread`. Per the deleted config comment, V1 "asserts on
multi-synchronizable-op kernels (CCL: device_synchronize + remote_store +
semaphore_wait)" — so **every CCL test likely regresses** UNLESS the new base's
3-pass umbrella now handles that case (which is the stated purpose of the
split-threads redesign). The two outcomes are mutually exclusive and the test
run will decide:
  - (a) umbrella does NOT handle CCL → restore the `be_opts` injection (V2).
  - (b) umbrella DOES handle CCL → V2 is obsolete; keep new default, fix docs.

The docs (CCL_SPEC.md, all_reduce_design.md, etc.) still reference
`config.use_split_unified_thread_v2`, which no longer exists — stale either way.

**FIX APPLIED:** restored both config fields (`config.py`, default True) and the
`be_opts` injection in `builder.py:_pipeline_passes()`, merged with the new
base's profiler-trace handling.

## R2 — TTKernelToEmitC.cpp build error: undeclared `funcOp` (HIGH, blocks build)

`lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp:3053` —
`patterns.add<TTKernelTensorAccessorDSpecOpRewriter>(typeConverter, funcOp.getContext())`.
Pre-rebase this lived in a function with `funcOp` in scope; the new base
refactored it into `buildPatterns(MLIRContext *context, ...)`. The rebase kept
the stale `funcOp.getContext()`.
**FIX APPLIED:** `funcOp.getContext()` → `context`.

## R3/R4 — TensorAccessor-DMA feature partially dropped in rebase (HIGH, blocks build)

The new base (`18908b135`) has no `useTensorAccessorDMA` anywhere; the branch
added it pre-rebase across pipeline + conversion-pass plumbing. The rebase merge
kept *some* uses but dropped the matching declarations, leaving dangling
references that don't compile (latent — ninja stopped at R2 before reaching
them):

- `D2MPipelines.cpp:268` used `options.useTensorAccessorDMA` but the
  `D2MPipelineOptions::useTensorAccessorDMA` declaration was gone. **FIX:**
  re-added the `Option<bool>` to `D2MPipelines.h`.
- pre-emitc pipeline no longer set `D2MToTTKernelOptions.useTensorAccessorDMA`.
  **FIX:** re-added the assignment in `D2MPipelines.cpp`.
- `ConvertD2MToTTKernel` pass lost its `use-tensor-accessor-dma` option in
  `Conversion/Passes.td`. **FIX:** re-added the `Option`.
- `populateD2MToTTKernelPatterns` body uses `useTensorAccessorDMA` (D2MToTTKernel
  .cpp:4320) but the parameter was dropped. **FIX:** re-added the param to decl
  (`.h`), definition (`.cpp`), the pass copy-ctor, and the call site in
  `D2MToTTKernelPass.cpp`.

## R5 — TTKernelOpsTypes.cpp: `appendArgImpl` numSlots vs sorted-insertion conflict (HIGH, blocks build)

True feature conflict. The **new base** introduced sorted insertion + dedup of
compile-time kernel args (`isLessArg`/`isSameArg`/`incrementCompileArgUsers`,
for CRTA determinism). The **branch** independently rewrote `appendArgImpl` to
reserve `numSlots` placeholder slots (`ArgType::Reserved`), needed so
TensorAccessor multi-slot CT args get correct flat uint32 offsets. The rebase
kept the new base's 3-arg body but the branch's 4-arg (`numSlots`) call sites
and header → `no matching function for call to 'appendArgImpl'`.

`ArgType::Reserved` is the highest enum value (8), so reserved placeholders do
not naturally sort adjacent to their primary arg — the two designs genuinely
interact.
**FIX APPLIED (merge):** restored the `numSlots` parameter and reservation, and
kept dedup + sorted insertion. The insertion scan skips `Reserved` placeholders
(so the sorted-range assumption holds), and `incrementCompileArgUsers` now
shifts downstream users by `numSlots` rather than 1.
**RESIDUAL RISK:** the merged ordering is only exercised once the build passes
and the d2m-jit/CRTA tests run; flagged for the arg-layout tests to confirm.

## R6 — D2MGenericRegionOps: `CoreReadOp`/`CoreWriteOp` op defs dropped (HIGH, blocks build)

ORIG_HEAD's `D2MGenericRegionOps.td` defined `D2M_CoreReadOp` and
`D2M_CoreWriteOp` immediately before `D2M_SynchronizedRegionOp`. The new base
removed `synchronized_region`; the rebase deleted all three as one block, but
the `.cpp` implementations of `CoreReadOp`/`CoreWriteOp` (verify/bufferize) and
their conversion rewriters survived → `use of undeclared identifier
'CoreReadOp'/'CoreWriteOp'`.
**FIX APPLIED:** re-added the two op definitions to the `.td` (after
`D2M_RemoteStoreOp`), NOT `synchronized_region` (its removal is intended).

## R7 — D2MToTTKernel.cpp: stale `createNocAsync*` arity + bogus `i32()` helper (HIGH, blocks build)

Two more rebase mismatches in the `CoreReadOp`/`CoreWriteOp` and TensorAccessor
lowerings:

- The new base added a `nocId` operand to `NocAsyncReadOp`/`NocAsyncWriteOp`
  (and a `materializeKernelNocId` helper, used by every other caller). The
  branch's `CoreReadOp`/`CoreWriteOp` rewriters predate it and called
  `createNocAsyncRead/Write` with 5 args. **FIX:** compute
  `nocId = materializeKernelNocId(...)` and pass it to the read/write and to the
  matching `NocAsync*BarrierOp` (so the barrier waits on the same NoC).
- TensorAccessor lowering called a nonexistent `i32(rewriter, loc, v)` helper
  (the codebase helper is `intConstant<int32_t>`; ORIG_HEAD used that). **FIX:**
  `i32(...)` → `intConstant<int32_t>(...)` at the 3 call sites.

## R8 — D2MToTTKernel.cpp: NocAsyncRead/WriteTileOp missing optional `noc` operand (HIGH, blocks build)

Same family as R7a: the new base added an optional `noc` operand to
`NocAsyncReadTileOp`/`NocAsyncWriteTileOp`. ODS-optional is not a C++ default
arg, so callers must pass it. The branch's TensorAccessor tile read/write
(D2MToTTKernel.cpp:3361/3364) omitted it.
**FIX:** pass `/*noc=*/Value()` (default NoC, matching the same function's
DPRINT probe and the pre-rebase behavior where the op had no `noc` operand).

## R9 — LowerDMAToFullyIndexedForm.cpp: `useTensorAccessorDMA` not threaded into patterns (HIGH, blocks build)

The `D2MLowerDMAReadToFullyIndexed`/`D2MLowerDMAWriteToFullyIndexed` rewriters
use `useTensorAccessorDMA` to skip plain shard-level DMAs (so the accessor path
handles them), but the rebase dropped the member + constructor param + the
pass's threading of the `use-tensor-accessor-dma` pass option.
**FIX:** restored the `useTensorAccessorDMA` ctor param/member on both rewriters
and pass it from the pass option at construction (matching ORIG_HEAD).

## R10 — builder.py: entire mesh/CCL/global_semaphore Python API dropped (CRITICAL, breaks all d2m-jit tests)

The biggest casualty. The rebase resolution of `tools/d2m-jit/_src/builder.py`
took the new base's version plus only a partial slice of the branch, dropping
~400 lines / 15+ functions: `GlobalSemaphore`/`global_semaphore`,
`mesh`/`MeshShard`/`mesh_shard`/`mesh_gather`/`_emit_mesh_shard`/
`_shard_logical_shape`/`_tensor_mesh_attr`, `reblock`, `reshape`, `arange`,
`fabric_config`, the mesh-device cache (`_get_cached_device`/
`_close_cached_device`), `_register_device`, and `_run_pipeline`. The leftover
`_execute` referenced `_get_cached_device` (gone) and `api.py`/`conftest.py`
import names that no longer existed → `ImportError: cannot import name
'GlobalSemaphore'` at collection time (every d2m-jit test fails to even load).

**FIX (proper 3-way merge):** reproduced the merge the rebase should have done —
`git merge-file` with base = old base (`9885724e9`), mine = branch (ORIG_HEAD),
theirs = new base (`18908b135`). 8 conflicts, all resolved by hand:
  - pipeline (`_build_pipeline`↔`_pipeline_passes`): merged → `_pipeline_passes`
    keeps the branch's be_opts injection (v2 + tensor-accessor) AND the new
    base's profiler-trace splice + config-driven `use-tile-matmul` + func.func
    wrap; `_run_pipeline` calls it.
  - branch CCL block + new base `reduction_layout`: kept both.
  - `_register_device`/`_run_pipeline` split (branch) vs combined (new base):
    kept the branch split, pointed it at `_pipeline_passes`.
  - device-cache block (branch) + `_maybe_enable_perf_trace` (new base): both.
  - `_emit_kernel_generic` / `CompiledKernel.__call__`: kept both new params
    (`fabric=` branch, `kernel_io_in_dram=` new base) and the new base's
    DRAM-output rebind alongside the branch's `is_view=False`.
Result: `import d2m_jit` succeeds; diff vs ORIG_HEAD is +140/-29 (exactly the
new base's additions). This supersedes the earlier in-place R1 edit.

## R8-revised — TensorAccessor tile read/write must materialize `noc` (HIGH, breaks all tensor-accessor-dma kernels)

After R8 made the build pass, the runtime pipeline failed: `failed to legalize
operation 'ttkernel.noc_async_read_tile'/'noc_async_write_tile'`. The new base's
TTKernel→EmitC lowering for these tile ops calls `ensureNocDeclaration`, which
*deliberately fails* on a null `noc` operand ("NoC operand is required ...
D2M-generated IR should materialize this operand explicitly"). My R8 stop-gap
`/*noc=*/Value()` therefore can't legalize.
**FIX:** pass `materializeKernelNocId(rewriter, op.getOperation())` (a static
NoC-index constant, as every other NocAsync caller does) to both tile ops.
This was the dominant runtime failure (eltwise/matmul/reductions/etc. all use
the default `use-tensor-accessor-dma=1` path).

## R11 — api.py: `@syntax("empty")` kernel-scope primitive dropped (HIGH, breaks CCL kernels)

The rebase merged the new base's `reduce_*` additions into the `zeros`→`matmul`
region of api.py and dropped the branch's `@syntax("empty")` registration
(`_empty_op`, the kernel-body `empty([...])` that emits `tensor.empty` for CCL
load buffers). Kernels using `empty(...)` failed at AST compile with `NameError:
unknown function 'empty' in kernel scope`.
**FIX:** re-added the `@syntax("empty")` block after `_zeros_op`. Also removed a
benign duplicate `@syntax("__matmul_acc__")` (identical merge artifact).

## d2m-jit test results — baseline + after fixes

Run per-file under a 300s timeout (`scratchpad/d2mjit_summary*.txt`). NOTE: the
known fabric bring-up timeout (multi-device CCL at device-open, see memory) is a
pre-existing environment issue, not a rebase regression. Non-fabric suites are
the regression signal.

After R8-revised + R11, the compute suites recovered (matmul 11/12, eltwise,
compare, broadcasts 16/16, bespoke, etc.). Two more Python regressions surfaced:

## R12 — api.py: `_shape_literal` reverted to literal-only (HIGH, breaks captured-int shapes)

The new base replaced the branch's capture-resolving `_shape_literal(node,
visitor)` (uses `visitor._eval_static_int`, so `empty([MT, KT])` / `zeros([M,
K])` work with closed-over int captures) with a literal-only
`_shape_literal(node)` (`_const_value`). Kernels generic over their block shape
(all CCL kernels) failed with "expected a Python literal ... got Name".
**FIX:** restored the 2-arg visitor form (`_eval_static_int` exists; the
`args_as_attr` resolver already dispatches 2-param callbacks).

## R13 — api.py: `_eltwise_block` lost its `out=` param (HIGH, breaks in-place accumulate)

`_eltwise_block` gained `preserve_reduced_axes=` (new base, reductions) but lost
the branch's `out=` (DPS init for in-place writes). `copy_` and `__add_acc__`
call it with `out=` → `TypeError: unexpected keyword argument 'out'` (loop
accumulators, in-loop output store). The docstring still described `out`.
**FIX:** merged both — re-added `out=None`, allocate `output = _as_value(out) if
out is not None else d2m.empty(output_ty)`, and use `output.type` as the generic
result type; kept `preserve_reduced_axes`. (Confirmed: test_loop_accumulator
4/4 passes after this.)

## R14 — api.py: `arange` dropped from the builder re-export (MEDIUM)

`arange` (host-side, restored to builder.py via R10) was no longer imported by
`api.py`, so `d2m.arange` raised `AttributeError` (test_arange_reshape).
**FIX:** re-added `arange` to the `from ._src.builder import (...)` block.

## R15 — Allocate.cpp: second remote_store walk still null-derefs (HIGH, SEGFAULT)

A pre-rebase commit (`4c6949391`) fixed a `d2m-allocate` segfault by changing
`isa<OperandAliasOp>(remoteStoreOp.getLocalBuffer().getDefiningOp())` to
`isa_and_nonnull<>` (the local buffer is a null-defining-op block arg for an
in-loop / loop-carried remote_store). The new base independently has **two**
such walks (the split-thread redesign added a speculative-stream-buffer walk);
the rebase applied the fix to the first but the second
(`analysis.generics` walk) kept the unguarded `isa<>` → segfault in
test_inloop_output_store (a lower-only regression guard for exactly this crash).
**FIX:** `isa<>` → `isa_and_nonnull<>` at the second site too (the line right
below already uses the null-safe `getDefiningOp<memref::AllocOp>()`).

---

# FINAL d2m-jit STATUS (after R1–R15)

Build: clean. `ttrt query`: clean. Per-file results on the fully-fixed tree:

PASSING (the entire non-CCL compute suite + one CCL test):
- eltwise, matmul (11 passed, 1 skipped), reductions (31), broadcasts (16),
  compare, ops, pattern_eltwise, views (10), tilize_untilize, round_trip (4),
  zeros_full_where, tensor_accessor_dma, arange_reshape (6, after R14),
  loop_accumulator (4, after R13), config, errors, bespoke.
- **all_gather_matmul (4/4 on device)** — CCL all-gather + TP matmul works.
- ccl_all_gather: skipped (intentional gate).

REMAINING FAILURES — one coherent cluster + the known fabric hang:

## R16 (FIXED) — fabric remote_store to a local-CB dest lost its get_write_ptr path

`convert-d2m-to-ttkernel`'s `D2MDMAWriteRewriter` for a cross-device fabric
write (`getStartDevice().size() > 0`) unconditionally called `buildNocEndpoint`
→ `castCBTypeAsAddress`, which emits an `unrealized_conversion_cast cb→i32` that
only resolves for genuinely remote/compile-time operands. For a fabric write
whose destination is a LOCAL CB scratch (the CCL tmp-buffer / loop-carried
output), that cast can't legalize → `failed to legalize
'builtin.unrealized_conversion_cast'`. ORIG_HEAD branched on `op.isDstRemote()`
and, for the local-CB case, used `GetWritePtrOp` (+ offset + the current core's
logical coords) for the destination address; the rebase dropped that `else`
branch.
**FIX:** restored the `isDstRemote()` branch (merged with the new base's `nocId`
threading). Confirmed: inloop_output_store, ring_all_reduce_loop,
all_reduce_attention, all_reduce_grid all pass (the CCL ones run on device).

## R17 (FIXED) — fcm not created for mesh_position-only threads (compile abort)

`getFabricConnectionManager` asserts the fcm exists. The rebase reverted the
pass's fcm-creation gate to "fabric write present" only. After
split-unified-thread, a local output store with a mesh_position-derived grid
index lands on a *different* NoC thread than the fabric send — that thread has
no fabric op, so no fcm was created, and `mesh_position`'s lowering aborted
(`Assertion 'fcm' failed`, D2MToTTKernel.cpp:182). ORIG_HEAD gated on `fcmUsers`
= fabric ops OR `mesh_position`.
**FIX:** broadened the gate in `D2MToTTKernelPass` to create the fcm when the
func has a fabric write/semaphore-inc/semaphore-set (startDevice) OR a
`MeshPositionOp`. Confirmed: meshpos_local_store passes; the rest of the cluster
still passes (no regression).

## (superseded) R16 original note — in-loop remote_store cb→i32 cast

`inloop_output_store`, `all_reduce_attention`, `all_reduce_grid`,
`ring_all_reduce_loop` all fail identically: `failed to legalize
'builtin.unrealized_conversion_cast' (!ttkernel.cb<..,tile> -> i32)`.
`meshpos_local_store` aborts (assertion) in the same lowering family.

Localized (via `scratchpad/repro_inloop.py`): the cast is introduced by pass #13
`d2m-to-ttkernel-pre-emitc-pipeline` (`convert-d2m-to-ttkernel`). It casts the
**output CB kernel argument** to `i32` when lowering a `remote_store(out, ...)`
that sits **inside an scf.for loop**. The non-loop equivalent
(`all_gather_matmul`'s `remote_store(out0, ...)`) lowers fine — so the loop
(loop-carried / repeated in-loop output store) is the trigger.

This is a genuine rebase regression (inloop_output_store is a lower-only guard
that verified cleanly at ORIG_HEAD per commit 4c6949391) but a DEEP one in the
D2MToTTKernel remote_store/DMA-write lowering interacting with the new base's
changes — NOT a mechanical dropped-code conflict like R1–R15. Left unfixed
deliberately: a speculative change to `convert-d2m-to-ttkernel` would risk the
now-green core suite (all of which goes through that pass). Needs a focused
diff of the in-loop remote_store lowering ORIG_HEAD vs now.

## R18 (FIXED) — TensorAccessor-DMA gave PCC 0.0 on DRAM operands

`test_simple::test_eltwise_dram` (a layout with `mem_space="dram"`) fails with
PCC 0.0 when `use_tensor_accessor_dma=True` (the branch default) and PASSES with
`D2M_JIT_USE_TENSOR_ACCESSOR_DMA=0`. This test is NEW from the base (absent at
ORIG_HEAD); the base ran it via fully-indexed DMA (it never had the
tensor-accessor-dma option), while the branch's TA-DMA work targeted L1 CCL
scratch buffers (its own `test_tensor_accessor_dma` passes). So this is a
TensorAccessor-DMA **DRAM-addressing** correctness gap newly exposed by combining
the base's DRAM test with the branch's TA-DMA default — not a mechanical
rebase-conflict like R1–R17. Workaround: `D2M_JIT_USE_TENSOR_ACCESSOR_DMA=0`.
**Root cause:** the TensorAccessor-DMA path is only validated for L1-resident
shards; on DRAM the accessor mis-addresses the paged/interleaved buffer. Two
layers were observed in the lowered IR: the row-major tilize/untilize staging
read (page = a 256-byte stick) AND the kernel's tile reads (page 4096) both
landed on the wrong DRAM pages, permuting the tiles (magnitudes correct, values
swapped). No passing test uses DRAM TA-DMA — every L1-default TA test
(tensor_accessor_dma, eltwise, all_gather_matmul, ring) is green — and the
fully-indexed path handles DRAM correctly (the test passes with TA-DMA off).
**FIX:** in `LowerDMAToFullyIndexedForm`, defer to the accessor path only for
L1-resident remote operands (`getSrcMemorySpace`/`getDstMemorySpace` ==
DeviceL1); DRAM remote operands lower to the proven fully-indexed form. Keeps
the validated L1 TA-DMA path unchanged. Confirmed: test_eltwise_dram (and all of
test_simple) passes; no regression on the L1 TA / CCL tests. (A future
enhancement could fix the DRAM accessor addressing itself and re-enable TA-DMA
for DRAM.)

## FINAL STATUS (after R1–R18)

The full d2m-jit suite is GREEN except `test_mesh` / `test_semaphore`, which hit
the pre-existing fabric bring-up hang (timeout) — not a rebase regression.
Every other file passes, including the previously-broken CCL loop-carried
cluster (all_gather_matmul, all_reduce_attention, all_reduce_grid,
ring_all_reduce_loop on device; inloop_output_store, meshpos_local_store
lower-only) and the DRAM eltwise (R18).

## Pre-existing (NOT a rebase regression) — fabric bring-up hang

`test_mesh` and `test_semaphore` time out at 300s. Per
[[blackhole-fabric-bringup-timeout]], multi-device fabric tests hang at
device-open on this box regardless of branch. (Some of their subtests may also
hit R16 at compile; the hang masks that.)
