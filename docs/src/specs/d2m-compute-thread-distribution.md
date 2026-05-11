# D2M Compute Thread Distribution

## Goal

Support a future hardware architecture that provides four FPU/SFPU compute
threads per L1/shard region. Every thread runs the same compiled program; the
only thread-specific value available at runtime is `my_thread_id` (an index in
`[0, 4)`). Conditions and loop bounds derived from `my_thread_id` are the only
allowed source of per-thread divergence.

The compiler needs a way to express "this work is distributed across N compute
threads" as a first-class IR construct, carry that abstraction through the D2M
backend, and lower it to single-program SPMD form only at the kernel-emission
boundary.

## Bring-up reference

The motivating workload is the large matmul in `test_steps.mlir`, produced by:

```
ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=$SYSTEM_DESC_PATH \
            matmul-interchange=2,0,1 use-tile-matmul=false enable-l1-acc=true" \
           ttir_module.mlir -o test_out.mlir --mlir-print-ir-after-all \
           &> test_steps.mlir
```

## Non-Goals

- Modeling cross-thread synchronization. The hardware threads execute the same
  program in lock-step within an L1 region; explicit synchronization is out of
  scope.
- General-purpose parallel reductions across compute threads. The first
  implementation distributes only parallel iterators of matmul-shaped ops.
- Replacing the existing `core_index` / multi-core distribution. Compute-thread
  distribution is one level *below* core distribution; both coexist.

## Why `scf.forall` (decision rationale)

The load-bearing decision is *not* to express per-thread ownership as
hand-inserted `scf.if` / `my_thread_id`-conditioned `scf.for` constructs
anywhere in the middle of the D2M backend:

1. **Distributed authorship.** Once guards live in normal SCF, every
   subsequent pass has to either understand the guard pattern or risk
   breaking it. The pattern is invisible at the IR level — nothing tells a
   pass "these conditionals encode parallelism, not generic control flow."
2. **No single materialization boundary.** With hand-inserted guards, the
   "where does SPMD start" question has no answer in the IR. With
   `scf.forall` + materialization pass, the boundary is a single op
   appearance and a single pass.
3. **Standard MLIR vocabulary.** `scf.forall` with a target-specific mapping
   attribute is exactly the construct GPU/SIMT codegen uses for the same
   purpose. Anyone reading the IR knows what it means.

## Empirical confirmation: tile-to-forall on memref works

A preliminary concern was whether MLIR's `tile_using_forall` machinery
requires tensor `shared_outs` semantics and therefore doesn't apply to
bufferized memref operands. **An empirical test with stock `mlir-opt
--transform-interpreter` confirmed it works on memref linalg ops.** A memref
`linalg.matmul` with `transform.structured.tile_using_forall ... num_threads
[4]` produces:

```mlir
scf.forall (%tid) in (4) {
  %off = affine.apply (d0 -> d0*16)(%tid)
  %sliceA = memref.subview %A[%off, 0][16, 32][1, 1] ...
  %sliceB = memref.subview %B[0, 0]   [32, 64][1, 1] ...
  %sliceC = memref.subview %C[%off, 0][16, 64][1, 1] ...
  linalg.matmul ins(%sliceA, %sliceB) outs(%sliceC)
} {mapping = [#gpu.thread<linear_dim_0>]}
```

This makes a second approach viable: tile the linalg op directly while it is
still alive, with the forall iterating *threads* and the body operating on
per-thread memref subviews — the standard GPU-style tile-and-distribute
shape.

## Two approaches

Both approaches share the same SPMD requirement at the kernel-emission
boundary, the same `d2m.my_thread_id` op, and the same `#d2m.compute_thread`
mapping attribute. They differ in (a) when the forall is introduced, (b)
what the forall's IV iterates, and (c) how it is materialized.

### Approach A — forall over work IV (late introduction, bounds materialization)

**Placement:** new pass `D2MDistributeComputeThreads` runs immediately after
`D2MLinalgToAffine` (linalg is gone, affine loop nest with `d2m.linalg_root`
exists), before `D2MOpScheduler`.

**Introduction IR:**
```mlir
scf.forall (%m) in (16) {
  // original per-tile compute body, indexed by %m
  // (acquire_dst / K-reduction / release_dst / store-back, etc.)
} {mapping = [#d2m.compute_thread<num = 4>]}
```
The IV iterates the *work*. Body is byte-for-byte the same as the
pre-distribution `affine.for` body — no subviews, no slice math.

**Materialization:**
```mlir
%tid    = d2m.my_thread_id : index
%begin  = arith.divui (M * %tid),         %num
%end    = arith.divui (M * (%tid + 1)),   %num
scf.for %m = %begin to %end step %c1 { /* unchanged body */ }
```
Trip count is thread-dependent (empty for threads whose `[begin, end)` is
zero).

**Loop selection** uses iterator-role tags (`d2m.iter_kind`,
`d2m.matmul_role`) attached by an extension of `D2MLinalgToAffine`.

**Pros:** the forall body never contains slicing math; the construct is the
only thing that changes between pre- and post-distribution IR; downstream
passes that walk loops see structurally unchanged inner bodies.

**Cons:** requires extending `D2MLinalgToAffine` with iterator-role tagging
(an extra cross-pass dependency); the loop-rewriting code that promotes
`affine.for` → `scf.forall` is hand-authored; the materialization pass
emits non-trivial arith for `[begin, end)`.

### Approach B — forall over threads via tile-and-distribute (early introduction, IV-substitution materialization)

**Placement:** new pass `D2MDistributeComputeThreads` runs immediately
*before* `D2MGenericTileComputeLoops`. Linalg ops are still live; DST-capacity
tiling has not yet happened.

**Mechanism:** uses MLIR's upstream `scf::tileUsingSCF` (or the equivalent
linalg-side transform) with `LoopType::ForallOp`, `num_threads = N` along the
chosen iterator, and `mapping = [#d2m.compute_thread<num = N>]`. MLIR
machinery produces all subviews and indexing math; we do not author them.

**Introduction IR:**
```mlir
scf.forall (%tid) in (4) {
  %off = affine.apply (d0 -> d0 * (M/4))(%tid)
  %sliceA = memref.subview %A[%off, 0]   [M/4, K] ...
  %sliceB = memref.subview %B[0, 0]      [K, N]   ...
  %sliceC = memref.subview %C[%off, 0]   [M/4, N] ...
  linalg.matmul ins(%sliceA, %sliceB) outs(%sliceC)
} {mapping = [#d2m.compute_thread<num = 4>]}
```
The IV iterates *threads*; the body is the per-thread work.

**Materialization:** substitute the forall IV with `d2m.my_thread_id` and
inline the body:
```mlir
%tid = d2m.my_thread_id : index
%off = affine.apply (d0 -> d0 * (M/4))(%tid)
%sliceA = memref.subview ...
...
linalg.matmul ins(...) outs(...)
```
No bounds arithmetic, no `scf.for` emission — just IV substitution and forall
erasure.

**Loop selection:** trivial — the linalg op is live, so M/N/K identification
is reading the linalg op's `iterator_types` and `indexing_maps`. No new
tagging needed.

**Pros:** all subview math is produced by upstream MLIR (not hand-authored);
no extension to `D2MLinalgToAffine`; materialization is the simplest possible
transformation (substitute + erase); the introduction-time IR shape exactly
matches the GPU/SIMT standard pattern; existing DST-capacity tiling
(`D2MGenericTileComputeLoops`) runs *inside* the per-thread forall body and
naturally tiles each thread's slice independently.

**Cons:** subview math and `affine.apply (d0 -> d0 * tile_size)` calls live
in the IR from introduction through materialization; downstream passes see
sliced memrefs, which is a wider IR diff than Approach A (though
semantically clean — these are normal memref subviews, not control-flow
guards).

## Comparison

| Axis | Approach A | Approach B |
|------|------------|------------|
| Introduction point | After `LinalgToAffine` | Before `GenericTileComputeLoops` |
| Forall IV iterates | Work (M tile indices) | Threads (`[0, num_threads)`) |
| Subviews in IR before materialization | No | Yes (per-thread slices) |
| Slice math authored by | (none) | Upstream MLIR tiling |
| Loop construct in body | `affine.for %m = 0 to M_per_thread` (after materialization) | Unchanged linalg op on subviewed operands |
| M/N identification | New iterator-role tags from `LinalgToAffine` | Linalg op live; introspect directly |
| Materialization rewrite | Bounds rewriting (`scf.for [begin, end)`) | IV substitution + forall erasure |
| Interaction with DST tiling | DST tiling runs before, sees affine nest unchanged | DST tiling runs *inside* per-thread forall body |
| Interaction with `InsertTileMatmulBlock` | Forall sits above `findComputeNestRoot` scope; structurally invisible | Same: forall sits above; matmul block insertion runs on per-thread sliced memrefs |
| New cross-pass dependency | Yes (iterator-role tags) | No |

Approach B trades "no slice math anywhere in early IR" (Approach A's headline
property) for "no hand-authored rewrite math anywhere" and a simpler
materialization. Both honor the "no hand-inserted guards" principle: subview
offsets inside a forall body with a mapping attribute are spatial
decomposition, not control-flow conditionals.

A detailed implementation plan for Approach B lives in
`docs/src/specs/d2m-compute-thread-tile-forall-plan.md`.

## Shared components

These are needed regardless of approach.

### `d2m.my_thread_id` op (and TTKernel counterpart)

A new index-producing op modeled on `D2M_CoreIndexOp`. It is a
`D2M_GenericRegionOp`, valid only inside a `d2m.generic` region — same
scoping contract as `core_index` and the other thread-context ops.

- TableGen: `include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td`
- Traits: `Pure`, `InferIntRangeInterface` returning `[0, num_compute_threads)`.
- Result: `Index`.
- Assembly: `d2m.my_thread_id : index`.
- Add a matching op to the TTKernel dialect and an entry in the D2M→TTKernel
  conversion that lowers `d2m.my_thread_id` to the kernel intrinsic. If the
  kernel-side intrinsic is not yet wired by the hardware bring-up branch, the
  lowering emits a placeholder call to a clearly-named symbol (e.g.
  `my_thread_id()`).

### `#d2m.compute_thread<num = N>` mapping attribute

A small TableGen-defined attribute conforming to
`DeviceMappingAttrInterface` so it can appear as `scf.forall`'s `mapping`
attribute.

- Parameter: `num : i64`.
- The materialization pass recognizes the forall by type identity of this
  attribute, not by interface-level mapping math.

### `D2MMaterializeComputeThreadForall` pass

A new D2M transform pass that lowers `scf.forall` with the
`#d2m.compute_thread` mapping to single-program SPMD form. The internal
rewrite differs between approaches (bounds rewriting for A; IV substitution
for B), but the pass exists and runs in the same pipeline position in both
cases: immediately before `D2MGenericRegionsToFuncs`. The func outliner sees
normal SPMD code in either case.

If any `#d2m.compute_thread` forall remains after this pass runs, fail.

### Pipeline options

`D2MPipelineOptions` gains:

- `enableComputeThreadTiling : bool = false`
- `numComputeThreads : i64 = 4`
- `computeThreadSplitDim : std::string = "m"`

Both passes (distribute + materialize) are gated on
`enableComputeThreadTiling`. Today's CI and the non-future-hardware path are
completely undisturbed.

## Structural facts (apply to both approaches)

- `D2MLinalgToAffine` already sets `d2m.linalg_root` on the root affine.for
  of every linalg-derived nest (`lib/Dialect/D2M/Transforms/LinalgToAffine.cpp`).
- `D2MInsertTileMatmulBlock::findComputeNestRoot`
  (`lib/Dialect/D2M/Transforms/InsertTileMatmulBlock.cpp`) climbs the
  `affine.for` parent chain starting at `d2m.tile_matmul` and stops as soon
  as the enclosing loop's body contains `acquire_dst`. The matmul block
  insertion never reaches an outer `scf.forall`.
- `d2m.my_thread_id` and `d2m.core_index` are siblings: `core_index` selects
  the physical core; `my_thread_id` selects which of N compute threads within
  that core.

## Downstream pass audit (applies to both approaches)

`scf.forall` is a `RegionBranchOpInterface` op whose body is a single block.
Most D2M passes walk regions via `Operation::walk` and treat region-owning
ops opaquely.

| Pass | Concern (Approach A) | Concern (Approach B) |
|------|---------------------|---------------------|
| `D2MGenericTileComputeLoops` | n/a (forall introduced after) | Forall introduced *before*; pass runs inside per-thread body. Verify it descends into forall regions. |
| `D2MLinalgToAffine` | n/a (forall introduced after) | Runs inside per-thread body; should be indifferent to the enclosing forall. |
| `D2MOpScheduler` | Operates on ops inside the compute region; the forall is an extra nesting level. | Same as A. |
| `D2MInsertSpillAndScratch` | Has its own `findLinalgRootLoop`. | Same as A. |
| `D2MInsertDstRegisterAccess` | Scans loop nests rooted at `d2m.linalg_root`. | Same. |
| `D2MInsertTileMatmulBlock` | `findComputeNestRoot` climbs the parent chain; non-`affine.for` parents stop the climb. | Same. |
| `D2MSFPUTileLoopFission` | Forall sits above the fission scope. | Same. |
| `D2MGenericLinearizeMemref` | Indifferent to forall. | Indifferent; sees sliced memrefs, which is its existing input shape. |
| `D2MGenericRegionsToFuncs` | Runs after materialization; no forall remains. | Same. |

The audit is bounded — for each pass, the change is either "no change" or
"one-line check that we don't pretend a non-affine parent is affine." Any
unexpected interaction surfaces when running the existing tests with
`enableComputeThreadTiling=true`.

## Risks and open items

- **TTKernel intrinsic readiness.** If the kernel runtime for the future
  hardware doesn't yet expose `my_thread_id()`, the conversion lowers to a
  named symbol that links to a stub. This is acceptable for compile-time
  bring-up and lets the kernel side catch up independently.
- **Non-matmul ops.** v1 explicitly skips them. Both approaches generalize
  cleanly; the heuristic for "which parallel dim to split" for arbitrary
  linalg shapes is out of scope for v1.
- **Multiple linalg roots inside one compute region.** Each gets its own
  forall in v1. If fused matmul + bias-add ends up here, the bias-add forall
  may distribute on a different dim than the matmul. Address only if
  encountered.
- **Constant trip-count assumption.** v1 requires the selected loop /
  iterator to have constant bounds.
- **Approach-A-specific:** iterator-role tagging adds a cross-pass coupling
  between `D2MLinalgToAffine` and the distribute pass.
- **Approach-B-specific:** running `D2MDistributeComputeThreads` *before*
  `D2MGenericTileComputeLoops` means DST-capacity tiling runs inside each
  forall iteration's body. Verify that the DST analysis sees per-thread
  output shapes (post-subview) correctly — it should, because it operates
  on the linalg op directly and the linalg op's operands are now subviews
  with the per-thread shape.

## Out-of-scope follow-ups

- Driving `num-compute-threads` from the system descriptor instead of a
  pipeline option.
- Generalizing distribution to elementwise and reduction generics.
- Allowing multiple `scf.forall` levels (distribute M *and* N across a 2D
  thread grid). The mapping-attribute scheme accommodates this.

## Alternatives considered and rejected

- **Hand-insert `scf.if (m >= begin && m < end)` guards inside the affine
  nest, early.** Distributed authorship; no IR-level signal that the guard
  encodes parallelism. Rejected.
- **Hand-insert `scf.for %m = %begin to %end` with `my_thread_id`-derived
  bounds, early.** Same authorship problem — `my_thread_id` and slice math
  leak into every pass's view of the IR. Rejected as the *introduction*
  form (this is exactly what Approach A's materialization pass emits at the
  end).
- **Fold the distribution into `D2MGenericTileComputeLoops` itself (no new
  pass).** Conditional multi-purpose passes are harder to read, test, and
  refactor; option-surface creep on a pass that already has a clear job.
  Approach B places a dedicated pass adjacent to it instead.
- **Split into four distinct compute regions, one per thread.** Violates the
  hardware requirement of identical thread code, and breaks the existing
  DMA/compute split contract. Rejected.
- **Use `scf.forall` but lower it immediately after creation.** Degenerates
  to "two passes that do what one pass would" and gives up the value of the
  explicit abstraction. Rejected.
- **Host the distribute logic in `D2MGenerateOuterLoops` (frontend,
  pre-bufferization).** Block-factor loops aren't the right granularity; the
  M axis ends up with three stacked levels of decomposition; the mapping
  attribute must survive bufferization. Rejected in favor of placing the
  pass adjacent to `D2MGenericTileComputeLoops` in the backend.
