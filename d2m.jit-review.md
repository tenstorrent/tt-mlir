# d2m.jit Frontend vs Main Pipeline: Comparative Review

## 1. Architecture: Where IR Comes From

| Aspect | Main Pipeline | d2m.jit |
|--------|--------------|---------|
| **Entry point** | TTIR ops (from torch-mlir or similar) | Python AST (`@jit`-decorated functions) |
| **IR generation** | `TTIRToD2M` pass converts TTIR → D2M | `mlir_generator.py` emits D2M directly from Python AST |
| **d2m.generic form** | Affine Blocked form (with `indexing_maps`, `iterator_types`, `block_factors`) | Explicit Datamovement form (empty `indexing_maps`/`iterator_types`/`block_factors=[]`) |
| **Grid selection** | `D2MGridSelection` pass (automatic) | User-specified `grid=(N,M)` in decorator |

**Insight**: The d2m.jit frontend skips the entire TTIR frontend (`TTMetalPipelines.cpp:75-123`) — roughly 20 passes including decomposition, TM explicitation, rank normalization, grid selection, and layout materialization. This is by design: the JIT user specifies grid, data layout, and access patterns directly in Python code. But it means **the IR entering the middleend is structurally different** — it's already in explicit datamovement form, bypassing the affine blocking → explicit form lowering chain.

## 2. Pass-by-Pass Middleend Comparison

### Main pipeline middleend order (`TTMetalPipelines.cpp:125-240`):

```
 1. D2MElementwiseFusion          ← SKIPPED in d2m.jit
 2. canonicalize                  ← SKIPPED
 3. TTIRBufferizationPipeline     ← ✓ (same)
 4. D2MAddScratchInputs           ← SKIPPED (replaced by Python convert_scratch_allocs)
 5. D2MGenericApplyInterchange    ← SKIPPED
 6. D2MGenerateOuterLoops         ← SKIPPED
 7. D2MAllocate                   ← SKIPPED
 8. D2MLowerMulticastLoads        ← SKIPPED
 9. D2MLowerToExplicitForm        ← SKIPPED (already in explicit form)
10. canonicalize + DecomposeMasking + DecomposeArange  ← SKIPPED
11. D2MGenericTileComputeLoops    ← ✓
12. D2MLinalgToAffine             ← ✓ (with use-tile-matmul=true)
13. D2MOpScheduler                ← SKIPPED
14. D2MSpillAndScratch            ← SKIPPED (Python lower_scratch_allocates replaces this)
15. canonicalize                  ← SKIPPED
16. D2MLowerScratchAllocate       ← SKIPPED (Python lower_scratch_allocates replaces this)
17. canonicalize                  ← SKIPPED
18. D2MInsertDstRegisterAccess    ← ✓
19. D2MSFPUTileLoopFission        ← ✓
20. canonicalize                  ← ✓
21. AffineLoopInvariantCodeMotion ← SKIPPED
22. lower-affine                  ← ✓
23. fold-memref-alias-ops         ← ✓
24. lower-affine                  ← ✓
25. D2MGenericLinearizeMemref     ← ✓
26. lower-affine                  ← ✓
27. D2MConvertLocalLoadStoreOpsToAliasedCBs  ← SKIPPED (fails with block_factors=[])
28. D2MLowerLoadStoreOpsToExplicitCBForm     ← ✓
29. D2MSplitUnifiedThread         ← ✓
30. D2MPreallocateMcastSemaphores ← ✓
31. D2MScheduleDMA                ← ✓
32. D2MLowerLoadStoreOpsToDMA     ← ✓
33. D2MLowerDMAToFullyIndexedForm ← ✓
34. Optimization passes (canonicalize, LICM, SCCP, CSE, IntRange, LICM)  ← only canonicalize
35. D2MGenericRegionsToFuncs      ← ✓
```

### d2m.jit middleend order (`ast_compiler.py`):

```
Phase 1 (pre_scratch_passes):
  bufferization → register-device → tile-compute-loops → linalg-to-affine →
  insert-dst → sfpu-fission → canonicalize → lower-affine → fold-memref-alias →
  lower-affine → explicit-cb-form

  [Python: convert_scratch_allocs]

Phase 2 (middle_passes):
  linearize-memref → lower-affine → split-unified-thread →
  preallocate-mcast-semaphores → schedule-dma → lower-load-store-to-dma →
  lower-dma-to-fully-indexed → canonicalize → regions-to-funcs

  [Python: lower_scratch_allocates]

Phase 3 (post_scratch_passes):
  convert-d2m-to-ttkernel → canonicalize → ttkernel-control-dst-section →
  canonicalize → convert-d2m-to-ttnn → ttkernel-hoist-inits
```

## 3. Key Differences and Their Implications

### A. Missing passes — functional gaps

| Skipped Pass | What It Does | Impact on d2m.jit |
|-------------|-------------|-------------------|
| **D2MElementwiseFusion** | Fuses chains of elementwise ops into single d2m.generic | Each `d2m.add`/`d2m.matmul` remains a separate linalg.generic inside the single d2m.generic. No cross-op fusion. |
| **D2MAddScratchInputs** | Adds a scratch CB operand to d2m.generic for spill | Replaced by Python `convert_scratch_allocs` + `lower_scratch_allocates`. |
| **D2MGenerateOuterLoops** | Creates outer affine loops over grid (blocked iteration) | d2m.jit has no outer loops — the single d2m.generic covers the whole grid with `block_factors=[]`. |
| **D2MAllocate** | L1 memory allocation, stream buffer management | No L1 allocation. CB sizes are implicit from tensor shapes. |
| **D2MOpScheduler** | Reorders ops for better DST register utilization | No scheduling — ops execute in source order. |
| **D2MSpillAndScratch** | Detects when intermediates need spilling to L1 | No spill analysis — user manually declares scratch via `alloc()`. |
| **D2MConvertLocalLoadStoreOpsToAliasedCBs** | CB aliasing optimization | Skipped because it fails on explicit form with `block_factors=[]`. |
| **Optimization passes** (LICM, SCCP, CSE, IntRange) | Standard compiler optimizations | Only canonicalize runs. No dead code elimination, loop hoisting, etc. |
| **ConvertTTKernelToEmitC** | Lowers to EmitC for code generation | Not run — d2m.jit stops at TTNN level. |

### B. Pass ordering differences

**Main: scratch before DST, JIT: scratch after DST**

Main pipeline:
```
AddScratchInputs → ... → SpillAndScratch → LowerScratchAllocate → InsertDstRegisterAccess
```

d2m.jit:
```
InsertDstRegisterAccess → ... → explicit-cb-form → [convert_scratch_allocs] → ... → regions-to-funcs → [lower_scratch_allocates]
```

In the main pipeline, `LowerScratchAllocate` runs **before** `InsertDstRegisterAccess` (line 194 vs 203). In d2m.jit, the scratch lowering runs **after** regions-to-funcs. This works because the d2m.jit scratch ops are opaque `d2m.scratch_allocate` ops that pass through unmodified until lowered.

**Main: aliased CBs before explicit-CB, JIT: skips aliased CBs entirely**

Main: `D2MConvertLocalLoadStoreOpsToAliasedCBs` → `D2MLowerLoadStoreOpsToExplicitCBForm`
JIT: `D2MLowerLoadStoreOpsToExplicitCBForm` only

This means d2m.jit creates **one CB per remote_load/store** with no aliasing optimization. In the main pipeline, aliased CBs allow multiple loads to share a single CB, reducing L1 pressure.

**Main: linearize before explicit-CB-form, JIT: explicit-CB before linearize**

Main: `lower-affine → fold-memref-alias → lower-affine → linearize → lower-affine → aliased-CBs → explicit-CB → split`
JIT: `lower-affine → fold-memref-alias → lower-affine → explicit-CB → [scratch_allocs] → linearize → lower-affine → split`

d2m.jit intentionally runs `explicit-cb-form` before `linearize-memref`. This is documented as necessary because linearize creates `collapse_shape` on local allocs without memory space, while explicit-cb-form replaces those allocs with `d2m.wait` results that have `#l1` memory space — doing it in the other order causes type mismatches.

### C. Scratch allocation: Python vs C++ approaches

**Main pipeline** (C++ `LowerScratchAllocate`):
- `D2MAddScratchInputs` adds a scratch CB operand to `d2m.generic`
- `D2MSpillAndScratch` inserts `d2m.scratch_allocate` for values needing spill
- `D2MLowerScratchAllocate` lowers scratch_allocate → `memref.subview` of scratch CB (2D `[1, total]` shape)
- Runs **inside** `d2m.generic` regions, before regions-to-funcs
- Uses `memref.expand_shape` to reshape flat subviews back to requested dimensions

**d2m.jit** (Python `convert_scratch_allocs` + `lower_scratch_allocates`):
- User explicitly calls `alloc()` for scratch buffers
- `convert_scratch_allocs` marks remaining `memref.alloc` as `d2m.scratch_allocate`
- `lower_scratch_allocates` consolidates into single flat CB (1D `[total]` shape)
- Runs **after** regions-to-funcs, in extracted compute kernel functions
- Avoids `memref.expand_shape` by directly replacing `collapse_shape` users with flat subview results (necessary because D2MToTTKernel has no pattern for `expand_shape`)

### D. IR generation form

The d2m.jit frontend generates IR in **explicit datamovement form** directly:
- `block_factors = []`, empty `indexing_maps` and `iterator_types`
- `d2m.remote_load`/`d2m.remote_store` with explicit grid coordinates
- `d2m.core_index` for core coordinates

This bypasses the entire **affine blocking → outer loops → explicit form** lowering chain in the main pipeline (passes 5-9 above). The main pipeline starts from a high-level declarative form and progressively lowers through:
```
Affine Blocked form → GenerateOuterLoops → Allocate → LowerToExplicitForm
```

## 4. Risks and Opportunities

### Risks

1. **No L1 allocation**: Without `D2MAllocate`, there's no guarantee that the generated CB sizes fit in L1. The user must ensure their tensor shapes and grid choices don't overflow L1 memory.

2. **No CB aliasing**: Every remote_load/store gets its own CB port. For programs with many loads, this wastes CB ports (hardware limit: 32) and L1.

3. **No op scheduling**: DST register pressure is unmanaged. For complex programs with many elementwise ops, the DST register file (16 tiles) could overflow.

4. **No spill analysis**: If the user doesn't declare enough scratch buffers, or declares them incorrectly, there's no safety net. The main pipeline's `SpillAndScratch` pass automatically handles this.

5. **Ordering fragility**: The d2m.jit pipeline has a non-trivial ordering dependency (explicit-cb-form must run before linearize) that's the opposite of the main pipeline. Any future change to either pipeline could break this.

### Opportunities

1. **Elementwise fusion**: Adding `D2MElementwiseFusion` before bufferization could reduce the number of separate linalg.generic ops, improving compute efficiency.

2. **Optimization passes**: The main pipeline runs LICM, SCCP, CSE, and IntRange optimizations after DMA lowering. Adding these to d2m.jit's `post_scratch_passes` (or between regions-to-funcs and D2MToTTKernel) would likely improve generated code quality.

3. **CB aliasing**: `D2MConvertLocalLoadStoreOpsToAliasedCBs` fails on `block_factors=[]`. Investigating why and potentially adapting it for explicit form could reduce CB port consumption.

4. **Use the C++ `d2m-lower-scratch-allocate` pass directly**: The current Python `lower_scratch_allocates` reimplements the C++ pass logic. If the d2m.jit pipeline were restructured to run scratch lowering before regions-to-funcs (matching the main pipeline order), it could use the C++ pass directly. This would require setting up `scratch_inputs` on the d2m.generic op (which `D2MAddScratchInputs` does in the main pipeline).

5. **EmitC backend**: d2m.jit stops at TTNN/TTKernel level. Extending to `ConvertTTKernelToEmitC` + `FormExpressions` would enable actual code generation.
