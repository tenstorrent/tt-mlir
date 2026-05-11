# Implementation plan — D2M compute-thread tile-and-distribute (Approach B)

Companion to `d2m-compute-thread-distribution.md`. This plan implements
**Approach B** from that document: forall over threads, introduced by a
dedicated pass placed immediately *before* `D2MGenericTileComputeLoops`, with
slicing produced by upstream MLIR tile-to-forall machinery and materialization
by IV substitution.

## Spike results (confirmed before implementation)

Run with stock `mlir-opt --transform-interpreter` from the toolchain and
`ttmlir-opt` parse/verify on hand-constructed IR:

1. **Matmul-shape `linalg.generic` on memref tiles via
   `transform.structured.tile_using_forall num_threads [4]`.** Produces
   `scf.forall (%tid) in (4) { affine.apply ; 3x memref.subview ;
   linalg.generic on subviews } {mapping = [...]}` — exactly the IR shape
   this plan assumes. Confirms `linalg.generic` (not just `linalg.matmul`)
   implements `TilingInterface` on bufferized operands and routes through
   `scf::tileUsingSCF` with `LoopType::ForallOp` correctly.
2. **Non-divisible trip count (M=15, num_threads=4).** Upstream tiling
   clamps the last thread's slice via `affine.min`; the tiled subview type
   becomes dynamic-rank (`memref<?x4xf32, ...>`). No custom handling needed.
3. **Fewer iterations than threads (M=2, num_threads=4).** Spare threads
   receive zero-sized subviews via `affine.max`/`affine.min` clamping; the
   linalg op on a zero-sized subview is a correct no-op.
4. **`scf.forall` inside `d2m.generic` compute region verifies.**
   Hand-constructed IR with the post-tile shape (d2m.generic body containing
   allocs + `scf.forall { subviews + linalg.generic { tile_matmul } }`)
   parses and verifies cleanly through `ttmlir-opt` — no D2M verifier
   rejects the new nesting.
5. **`!ttcore.tile<32x32, bf16>` element types compose with subviews and
   linalg ops.** Hand-constructed test above uses these types; the existing
   `D2MGenericTileComputeLoops` already tiles linalg ops with this element
   type via `linalg::tileLinalgOp` (same `TilingInterface` infrastructure).
6. **Branch hygiene clean.** `git log --all --grep` for `my_thread_id`,
   `compute_thread`, `MyThreadId`, and "compute thread" finds no in-flight
   parallel work that would conflict.

### Pipeline-context observation

At the proposed pass placement (post-`DecomposeArange`, pre-`GenericTileComputeLoops`),
the matmul `linalg.generic` lives inside the d2m.generic compute region, inside
an outer `scf.for` whose body is one grid-synchronized mcast round:

```
d2m.generic { ... unified thread ...
  scf.for %round = 0 to 64 {
    %A = d2m.remote_load ... mcore[...] mshape[1, 8]   // mcast A-slice across a row
    %B = d2m.remote_load ... mcore[...] mshape[8, 1]   // mcast B-slice across a column
    linalg.generic { matmul-shape }                    // <- pass tiles this
    d2m.remote_store ...
  } {d2m.blocking_loop = 0}
}
```

The outer loop is *not* serial per-core in any meaningful sense — each
iteration is a grid-parallel mcast round in which all 64 cores receive their
slice and do one matmul tile accumulation. The serial work that
compute-thread distribution divides is the *within-round* compute on each
core: an 8x8 tile-matmul accumulation, partitioned across 4 compute threads
by M.

Tiling the inner `linalg.generic` by M with `num_threads=4` places the
produced `scf.forall` inside the round loop. After `SplitUnifiedThread`
later in the pipeline, the loads/stores end up on the data-movement thread
and the forall body ends up on the compute side, where it expresses
"per-round work for one of N compute threads."

Cross-thread synchronization (how multiple compute threads coordinate with
data movement on the future hardware) is **out of scope** for this work and
is being addressed in a different branch by introducing new primitives. The
distribute pass emits the same CB / semaphore op vocabulary as today; the
runtime / new primitives interpret it.

The spike confirms feasibility and the correct semantic placement.
Implementation can proceed step-by-step per the sections below.

## Pipeline placement

```
D2MDecomposeMasking
D2MDecomposeArange
└─► [NEW]  D2MDistributeComputeThreads        // gated by enableComputeThreadTiling
D2MGenericTileComputeLoops                    // runs INSIDE each per-thread forall body
D2MLinalgToAffine
D2MOpScheduler
D2MInsertSpillAndScratch
canonicalize
D2MLowerScratchAllocate
canonicalize
D2MInsertDstRegisterAccess
D2MInsertTileMatmulBlock
D2MSFPUTileLoopFission
...
└─► [NEW]  D2MMaterializeComputeThreadForall  // gated by enableComputeThreadTiling
D2MGenericRegionsToFuncs
```

Both new passes are gated on `enableComputeThreadTiling` and absent when the
flag is off — current CI is unchanged.

### Why "separate pass before `D2MGenericTileComputeLoops`" rather than inline

- **Single responsibility per pass.** `D2MGenericTileComputeLoops` keeps its
  one job (DST-capacity tiling). The new pass has one job (thread
  distribution).
- **No conditional logic inside `D2MGenericTileComputeLoops`.** No flag-gated
  code path that future readers have to mentally execute alongside the
  pass's primary purpose.
- **Independent testability.** Unit tests for either pass need only that
  pass running, not the other.
- **Order is the contract.** Running thread distribution first means DST
  capacity tiling automatically operates per-thread — each thread's linalg
  op already has the per-thread output shape (post-subview), so the existing
  `calculateOutputSubblockFactors` math operates on the slice shape, which
  is exactly what we want.

## Component 1 — `d2m.my_thread_id` op

**File:** `include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td`

Add immediately after `D2M_CoreIndexOp` (line 2886):

```tablegen
def D2M_MyThreadIdOp : D2M_GenericRegionOp<"my_thread_id",
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
    let summary = "Compute thread id within an L1/shard region.";
    let description = [{
      Return the compute thread index, in [0, num_compute_threads). On the
      target hardware multiple compute threads execute the same program
      within one L1 region; this op returns the per-thread identifier so
      thread-specific behavior can be derived from it.

      Only legal inside a d2m.generic compute region.
    }];

    let results = (outs Index:$result);
    let assemblyFormat = [{ attr-dict `:` type($result) }];
    let hasVerifier = 1;
}
```

**C++ side:** mirror `D2M_CoreIndexOp`'s verifier (must be inside a
`d2m.generic` compute region) and `inferResultRanges` (returns
`[0, num_compute_threads)` — for v1, hard-code `4`; a follow-up reads it from
a pipeline option or the system descriptor).

**TTKernel counterpart:** add a corresponding op to the TTKernel dialect TD
and a conversion in `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` that
emits the kernel intrinsic. For v1, the intrinsic name is `my_thread_id()`;
if the kernel runtime hasn't wired it yet, this lowers to a symbol stub the
runtime team fills in.

**Tests:**
- `test/ttmlir/Dialect/D2M/IR/my_thread_id.mlir` — round-trip parse/print.
- `test/ttmlir/Conversion/D2MToTTKernel/my_thread_id_lowering.mlir` —
  verifies the conversion produces the TTKernel op.

## Component 2 — `#d2m.compute_thread<num = N>` mapping attribute

**File:** `include/ttmlir/Dialect/D2M/IR/D2MAttrs.td` (or whichever D2M attrs
file exists; verify at implementation time). The attribute must conform to
the upstream `DeviceMappingAttrInterface` declared in
`mlir/Interfaces/DeviceMappingInterface.h`.

```tablegen
def D2M_ComputeThreadMappingAttr : AttrDef<D2M_Dialect, "ComputeThreadMapping", [
    DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>
  ]> {
  let mnemonic = "compute_thread";
  let parameters = (ins "int64_t":$num);
  let assemblyFormat = "`<` `num` `=` $num `>`";
}
```

Interface methods can be minimal (a unique `mappingId`, `isLinearMapping()`
returning true). The materialization pass identifies forall ops by checking
that *any* element of the `mapping` array is of this attribute type.

**Test:** parse/print round-trip via a small MLIR file with an `scf.forall`
carrying the attribute.

## Component 3 — `D2MDistributeComputeThreads` pass (the new pass)

**Files:**
- `include/ttmlir/Dialect/D2M/Transforms/Passes.td` — pass declaration.
- `lib/Dialect/D2M/Transforms/DistributeComputeThreads.cpp` — implementation.
- `lib/Dialect/D2M/Transforms/CMakeLists.txt` — add the new .cpp.

**Pass declaration (Passes.td):**

```tablegen
def D2MDistributeComputeThreads : Pass<"d2m-distribute-compute-threads", "ModuleOp"> {
  let summary = "Distribute matmul-shaped linalg ops across compute threads via scf.forall.";
  let description = [{
    Tile a chosen iterator of matmul-shaped linalg.generic ops by num_threads,
    producing scf.forall with #d2m.compute_thread mapping. The forall iterates
    threads; the body operates on per-thread memref subviews.
  }];
  let options = [
    Option<"numComputeThreads", "num-compute-threads", "int64_t", "4",
           "Number of compute threads per L1 region.">,
    Option<"splitDim", "split-dim", "std::string", "\"m\"",
           "Which matmul iterator to distribute: m or n.">,
  ];
  let dependentDialects = [
    "scf::SCFDialect",
    "affine::AffineDialect",
    "memref::MemRefDialect",
    "linalg::LinalgDialect",
  ];
}
```

**Algorithm:**

```
runOnOperation():
  ModuleOp moduleOp = getOperation()
  walk every d2m::GenericOp:
    if generic is in explicit-datamovement form: skip   (mirror GenerateOuterLoops gating)
    find all linalg.generic in the compute region
    if exactly one linalg op AND it is matmul-shaped:
      distribute it
    else:
      skip (v1 limitation; record as follow-up)

distribute(linalg::GenericOp op):
  // Identify which iteration-space dim corresponds to the requested split.
  auto outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0))
  int64_t mDim = outputMap.getResult(0).cast<AffineDimExpr>().getPosition()
  int64_t nDim = outputMap.getResult(1).cast<AffineDimExpr>().getPosition()
  int64_t targetDim = (splitDim == "n") ? nDim : mDim

  // Build SCFTilingOptions: tile only targetDim by num_threads.
  unsigned numLoops = op.getNumLoops()
  SmallVector<OpFoldResult> numThreads(numLoops, IntegerAttr 0)
  numThreads[targetDim] = IntegerAttr(numComputeThreads)

  scf::SCFTilingOptions opts
  opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp)
  opts.setNumThreads(numThreads)
  opts.setMapping({ D2M_ComputeThreadMappingAttr::get(ctx, numComputeThreads) })

  // Invoke MLIR machinery.
  auto result = scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), opts)
  if failed: signalPassFailure()
```

**Matmul detection (v1 heuristic):**

```
isMatmulShaped(linalg::GenericOp op):
  if op.getNumLoops() != 3: return false
  iterTypes = op.getIteratorTypesArray()
  countParallel = count(iterTypes == parallel)
  countReduction = count(iterTypes == reduction)
  if countParallel != 2 || countReduction != 1: return false
  if op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1: return false
  return true
```

This is enough for the bring-up matmul. Tightening (e.g., explicit
`linalg::isaMatmulOpInterface` check) can land as a follow-up.

**Output IR shape** (per the empirical test):

```mlir
scf.forall (%tid) in (4) {
  %off = affine.apply (d0 -> d0 * tileSize)(%tid)
  %sliceA = memref.subview %A[%off, 0]   [tileSize, K] [1, 1]
  %sliceB = memref.subview %B[0, 0]      [K, N]        [1, 1]
  %sliceC = memref.subview %C[%off, 0]   [tileSize, N] [1, 1]
  linalg.generic { ... matmul-shape indexing maps ... }
                 ins(%sliceA, %sliceB) outs(%sliceC) { ... }
} {mapping = [#d2m.compute_thread<num = 4>]}
```

`tileSize` is `ceildiv(M, num_threads)` — MLIR computes this for us when we
use `setNumThreads`.

**Tests** (`test/ttmlir/Dialect/D2M/Transforms/`):

- `distribute_compute_threads_m.mlir` — matmul d2m.generic, split-dim=m;
  CHECK lines verify the scf.forall, mapping attribute, and three
  memref.subview ops.
- `distribute_compute_threads_n.mlir` — same with split-dim=n.
- `distribute_compute_threads_non_matmul_skipped.mlir` — elementwise
  d2m.generic; CHECK lines verify no scf.forall is introduced.
- `distribute_compute_threads_non_divisible.mlir` — matmul with
  `M=15, num_threads=4`; CHECK lines verify the forall is created and the
  per-thread subview size math handles the boundary (MLIR emits a min/affine
  expression for the last tile's size).

## Component 4 — `D2MMaterializeComputeThreadForall` pass

**Files:**
- `include/ttmlir/Dialect/D2M/Transforms/Passes.td` — pass declaration.
- `lib/Dialect/D2M/Transforms/MaterializeComputeThreadForall.cpp` —
  implementation.

**Pass declaration:**

```tablegen
def D2MMaterializeComputeThreadForall :
    Pass<"d2m-materialize-compute-thread-forall", "ModuleOp"> {
  let summary = "Lower #d2m.compute_thread scf.forall to single-program SPMD form.";
  let description = [{
    For each scf.forall whose mapping contains #d2m.compute_thread, insert a
    d2m.my_thread_id op, substitute it for the forall's induction variable,
    inline the body into the parent block, and erase the forall.

    Fails if any #d2m.compute_thread forall remains after the pass.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}
```

**Algorithm:**

```
runOnOperation():
  walk every scf::ForallOp:
    if any mapping element is D2M_ComputeThreadMappingAttr:
      lower(forall)

lower(scf::ForallOp forall):
  rewriter.setInsertionPoint(forall)
  Value tid = rewriter.create<d2m::MyThreadIdOp>(loc)

  // Replace the single IV with tid.
  Block *body = forall.getBody()
  assert(forall.getInductionVars().size() == 1)
  body->getArgument(0).replaceAllUsesWith(tid)

  // Drop the in_parallel terminator (must be empty for memref forall).
  Operation *terminator = body->getTerminator()
  assert(terminator->getNumOperands() == 0)
  rewriter.eraseOp(terminator)

  // Inline body into the parent block, right after `forall`.
  rewriter.inlineBlockBefore(body, forall, /*argValues=*/{})
  rewriter.eraseOp(forall)
```

**Output IR shape:**

```mlir
%tid = d2m.my_thread_id : index
%off = affine.apply (d0 -> d0 * tileSize)(%tid)
%sliceA = memref.subview %A[%off, 0]   [tileSize, K] [1, 1]
%sliceB = memref.subview %B[0, 0]      [K, N]        [1, 1]
%sliceC = memref.subview %C[%off, 0]   [tileSize, N] [1, 1]
// ...whatever downstream passes left in the body (post-DST tiling,
// affine.for nest, tile_matmul_block, etc.)...
```

**Tests:**

- `materialize_compute_thread_forall_basic.mlir` — hand-written input with
  an `scf.forall (%tid) in (4) { ... } {mapping=[#d2m.compute_thread<num=4>]}`
  enclosing two memref.subview ops and a simple op using `%tid`. CHECK lines
  verify the forall is gone, `d2m.my_thread_id` appears, and all uses of the
  former IV now use the `my_thread_id` result.
- `materialize_compute_thread_forall_multiple.mlir` — two forall ops in
  different functions; both are lowered.
- `materialize_compute_thread_forall_unrelated_forall.mlir` — an
  `scf.forall` *without* the `#d2m.compute_thread` mapping (e.g., a generic
  one); CHECK lines verify it is *not* touched.

## Component 5 — Pipeline wiring

**Files:**
- `include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h` — add options.
- `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` — add passes.

**Options on `D2MPipelineOptions`:**

```cpp
PassOptions::Option<bool> enableComputeThreadTiling{
    *this, "enable-compute-thread-tiling",
    llvm::cl::desc("Distribute matmul work across compute threads."),
    llvm::cl::init(false)};
PassOptions::Option<int64_t> numComputeThreads{
    *this, "num-compute-threads",
    llvm::cl::desc("Number of compute threads per L1 region."),
    llvm::cl::init(4)};
PassOptions::Option<std::string> computeThreadSplitDim{
    *this, "compute-thread-split-dim",
    llvm::cl::desc("Which matmul iterator to distribute: m or n."),
    llvm::cl::init("m")};
```

**Wiring in `createD2MBackendPipeline`:**

```cpp
// Existing:
pm.addPass(d2m::createD2MDecomposeMasking());
pm.addPass(d2m::createD2MDecomposeArange());

// NEW: distribute before DST-capacity tiling so that pass operates per-thread.
if (options.enableComputeThreadTiling) {
  d2m::D2MDistributeComputeThreadsOptions distOptions;
  distOptions.numComputeThreads = options.numComputeThreads;
  distOptions.splitDim = options.computeThreadSplitDim;
  pm.addPass(d2m::createD2MDistributeComputeThreads(distOptions));
}

pm.addPass(d2m::createD2MGenericTileComputeLoops(...));
// ... existing passes ...

// ... near the end of the backend pipeline, before generic-regions-to-funcs:
if (options.enableComputeThreadTiling) {
  pm.addPass(d2m::createD2MMaterializeComputeThreadForall());
}

pm.addPass(d2m::createD2MGenericRegionsToFuncs());
```

## Component 6 — Downstream pass interactions to verify

These are pass-level "should-work-but-confirm" items. Each becomes a
verification step in the implementation order below.

| Pass | What runs inside the forall body | Verification |
|------|----------------------------------|--------------|
| `D2MGenericTileComputeLoops` | DST-capacity tiling on per-thread sliced linalg op | The linalg op's output shape is now the per-thread slice; `calculateOutputSubblockFactors` should produce sensible tile sizes. Run pipeline test, compare against current shape divided by num_threads on the split dim. |
| `D2MLinalgToAffine` | Linalg → affine.for nest, inside forall body | Indifferent to enclosing op. No change expected. |
| `D2MOpScheduler` | Scheduler ops in the compute region | Walks regions; forall is just an extra nesting level. Confirm scheduler descends. |
| `D2MInsertSpillAndScratch` | Anchors on `d2m.linalg_root` | Tag is preserved; the anchor is inside the forall body. Confirm parent walks tolerate the forall. |
| `D2MInsertDstRegisterAccess` | Acquire/release insertion around acquire-bearing loops | Forall is above any acquire-bearing structure; should be invisible to this pass. |
| `D2MInsertTileMatmulBlock` | Replaces inner affine nest with `tile_matmul_block` | `findComputeNestRoot` stops at non-affine parents; the forall is non-affine and outside the matmul block scope. Confirm with an IR-after-all check. |
| `D2MSplitUnifiedThread` | Partitions the unified d2m.generic body into compute-thread and data-movement-thread bodies, based on op affinity. | The forall body contains compute ops; surrounding loads/stores are data movement. Verify the pass routes the forall wholesale onto the compute side without descending into it. If it tries to split per-op inside the forall body, it will produce broken IR — teach it to treat the forall as an opaque compute-side block. |

If any of these requires real surgery, the plan grows; record findings as
they arise.

## Component 7 — Pipeline-level test

**File:** `test/ttmlir/Dialect/D2M/Pipelines/compute_thread_pipeline.mlir`

A minimal TTIR matmul module, lowered via:

```
ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=$SYSTEM_DESC_PATH \
            matmul-interchange=2,0,1 use-tile-matmul=false enable-l1-acc=true \
            enable-compute-thread-tiling=true num-compute-threads=4 \
            compute-thread-split-dim=m"
```

CHECK lines verify:

1. `scf.forall` with `#d2m.compute_thread` appears after
   `D2MDistributeComputeThreads`.
2. The forall survives through `D2MGenericTileComputeLoops`, `D2MLinalgToAffine`,
   and `D2MInsertTileMatmulBlock`.
3. `d2m.my_thread_id` appears in the output of
   `D2MMaterializeComputeThreadForall`.
4. No `scf.forall` (with our mapping) remains after the materialization pass.
5. The final TTKernel-stage output has one `my_thread_id` op per distributed
   forall.

## Testing principle

Every step that lands a new op, attribute, or pass must include at least one
lit-style IR test verifying the new behavior — round-trip for ops and
attributes, before/after IR checks for passes. Tests live in
`test/ttmlir/Dialect/D2M/` mirroring the existing layout. The intermediate
commits in the implementation order below each land their pass *and* their
test together — no commit introduces a pass without coverage of its visible
behavior. The pipeline-level test at step 6 is in addition to (not a
replacement for) the per-pass unit tests.

## Implementation order

TDD-friendly sequencing. Each step lands behind the
`enableComputeThreadTiling=false` default; existing tests continue to pass at
every step.

1. **`d2m.my_thread_id` op (TableGen + verifier + range inference).** Round-trip test.
2. **`#d2m.compute_thread` mapping attribute.** Round-trip test.
3. **`D2MMaterializeComputeThreadForall` pass.** Hand-written input MLIR;
   unit tests as in Component 4.
4. **`D2MDistributeComputeThreads` pass.** Hand-written matmul-shaped input
   MLIR; unit tests as in Component 3. This step verifies that
   `scf::tileUsingSCF` on a `linalg.generic` with matmul-shape indexing maps
   (not just `linalg.matmul`) produces the expected forall + subviews.
5. **TTKernel-side `my_thread_id` op + D2M→TTKernel conversion.** Conversion
   unit test.
6. **Pipeline wiring + pipeline-level test (Component 7).** First run that
   composes both passes through the full backend.
7. **Downstream-pass audit (Component 6).** Run the existing matmul tests
   with `enable-compute-thread-tiling=true` and triage any failures.

## Edge cases and limitations (v1)

- **Single linalg op per d2m.generic compute region.** v1 only distributes
  when the region contains exactly one matmul-shaped linalg.generic. Regions
  with bias-add or other companion ops are skipped (the matmul is *not*
  distributed in that case). A follow-up extends to fused regions by either
  wrapping all sibling ops in the same forall or by integrating with
  existing element-wise-fusion infrastructure.
- **Non-divisible trip counts.** MLIR's tile-using-forall handles boundary
  slices via affine min expressions. v1 just trusts this; the
  non-divisible test (Component 3) is the verification.
- **`num_threads > trip_count`.** Some threads get empty slices. v1 lets
  MLIR emit whatever it emits and verifies the test suite passes on a
  representative shape; if MLIR's behavior is awkward, we add an early-out
  in the distribute pass that skips distribution when
  `trip_count < num_threads`.
- **Matmul detection heuristic.** v1 uses the simple "3 loops, 2 parallel +
  1 reduction, 2 inputs + 1 output" check. Tightening to a real
  `isaMatmulOpInterface` check is a follow-up.
- **TTKernel intrinsic.** If the kernel runtime hasn't wired `my_thread_id()`
  yet, the conversion targets a placeholder symbol; the rest of the
  compiler pipeline does not block on this.

## External dependency (not introduced by this work)

- **`my_thread_id()` kernel intrinsic.** The TTKernel-side op produced by
  `D2MMyThreadIdOp` lowering must resolve to a real per-thread identifier
  at runtime. If the runtime exposes it under a different name, only the
  conversion mapping changes — the compiler IR shape is unaffected.

Cross-thread synchronization (CB handshakes, semaphore-wait-for-N-threads,
etc.) is being handled entirely on a separate branch via new primitives
and is **explicitly out of scope** for this work. This plan does not
introduce, audit, or emit anything related to multi-thread synchronization.

## Open items (track during implementation)

Closed by the spike (see "Spike results" above): `linalg.generic` /
`linalg.matmul` parity, `linalg.generic` implementing `TilingInterface` on
memref, non-divisible / fewer-than-threads edge cases, scf.forall legality
inside d2m.generic.

Remaining:

- Confirm whether the D2M attrs TableGen file exists; if not, choose a
  sensible location (`include/ttmlir/Dialect/D2M/IR/D2MAttrs.td`) and add it
  to the build. (Quick `find` at implementation time.)
- Verify the in-parallel terminator handling in the materialization rewrite
  on a real post-tile forall: the spike showed no `tensor.parallel_insert_slice`
  ops in the body, but the terminator op (`scf.in_parallel`) itself may
  still be present — the rewrite handles this by erasing it.
- Decide whether to lower `d2m.my_thread_id` to a real TTKernel intrinsic or
  a placeholder for v1, depending on what the future-hardware runtime
  exposes when implementation starts.
