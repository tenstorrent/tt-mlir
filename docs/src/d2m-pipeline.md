# The D2M Pipeline

This page is a reference for the D2M ("Direct-to-Metal") compiler pipeline:
the chain of MLIR passes that lowers `d2m.generic` ops on tensors all the way
down to TTKernel/TTMetal/TTNN IR and EmitC. It documents each stage, the IR
contract between stages, the memory/memref types that flow through the
pipeline, the synchronization model, and the `ttnn-mode` variant.

The scope of this doc starts at `ttir-to-d2m`. The TTIR normalization passes
that run before it (decomposition, broadcast folding, rank normalization,
canonicalization, etc.) are part of the same `D2MFrontendPipeline` but are
upstream of D2M and are intentionally treated as a black box here. See
`include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h` and
`lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` for the full list.

## 1. Pipeline Overview

The pipeline is assembled in `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` and
exposes the following named entry points (registered in `registerD2MPipelines()`):

| Pipeline | Factory | What it does |
|---|---|---|
| `--ttir-to-ttmetal-pipeline` | `createTTIRToTTMetalPipeline` | End-to-end: wraps the function in a DeviceModule, then runs FE → BE → D2M→TTKernel pre-EmitC → (D2M→TTMetal or D2M→TTNN) → TTKernel finalization → EmitC |
| `--d2m-fe-pipeline` | `createD2MFrontendPipeline` | TTIR normalization → `ttir-to-d2m` → grid selection → layout lowering → bufferization → allocation → explicit datamovement form |
| `--d2m-be-pipeline` | `createD2MBackendPipeline` | Tile compute lowering, DST register management, affine flattening, DMA scheduling, region-to-func hoisting |
| `--d2m-to-ttkernel-pipeline` | `createD2MToTTKernelPipeline` | D2M → TTKernel + TTKernel finalization + EmitC |
| `--d2m-to-ttmetal-pipeline` | `createD2MToTTMetalPipeline` | `convert-d2m-to-ttmetal` (dispatch-level wrap) |
| `--d2m-to-ttnn-pipeline` | `createD2MToTTNNPipeline` | `convert-d2m-to-ttnn` (dispatch-level wrap for the TTNN backend) |
| `--ttir-bufferization-pipeline` | `createTTIRBufferizationPipeline` | One-shot bufferization (TTCore-aware by default; standard MLIR when `ttnn-mode=true`) |

The options struct that drives all of these is `D2MPipelineOptions`
(`include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h:15-214`). Key flags:
`mesh-shape`, `mesh-topology`, `system-desc-path`, `mock-system-desc-arch`,
`max-dst-physical-size-tiles`, `enable-elementwise-fusion`,
`enable-eltwise-reduction-fusion`, `matmul-interchange`, `use-tile-matmul`,
`collapse-tensors-2d`, `default-input-memspace`, `default-output-memspace`,
`disable-tolayout-folding`, `insert-profiler-traces`, `set-math-fidelity`,
`num-stream-buffers`, `allow-l1-output-spilling`, `stream-insert-policy`,
`available-l1-addr-range`, `ttnn-mode`, `global-data-format-target`,
`enable-op-scheduler`, `enable-multicast-inference`, `enable-l1-acc`.

### Pipeline flow at a glance

```
                              ┌────────────────────────────────────────────┐
TTIR (named ops, tensors)  ─► │ TTIR normalization prelude (out of scope)  │
                              └────────────────────────────────────────────┘
                                                │
                                  ttir-to-d2m   ▼
                              ┌────────────────────────────────────────────┐
D2M on tensors             ─► │ D2M Frontend: grid sel, layout lowering,   │
                              │   bufferization, allocate, explicit form   │
                              └────────────────────────────────────────────┘
                                                │
                                                ▼
                              ┌────────────────────────────────────────────┐
D2M on memrefs (explicit)  ─► │ D2M Backend: tile compute, DST, affine,    │
                              │   DMA split + schedule, regions-to-funcs   │
                              └────────────────────────────────────────────┘
                                                │
                          convert-d2m-to-ttkernel▼
                              ┌────────────────────────────────────────────┐
TTKernel functions         ─► │ TTKernel finalize: control-dst, hoist init │
                              └────────────────────────────────────────────┘
                                                │
                                                ▼
                          ┌───────────────────────────────────────┐
                          │  ttnn-mode=false: convert-d2m-to-ttmetal │
                          │  ttnn-mode=true:  convert-d2m-to-ttnn    │
                          └───────────────────────────────────────┘
                                                │
                              convert-ttkernel-to-emitc + emitc-form-expressions
                                                ▼
                                       TTMetal / TTNN flatbuffer
```

## 2. The Frontend Pipeline (from `ttir-to-d2m` downward)

`createD2MFrontendPipeline` lives at
`lib/Dialect/D2M/Pipelines/D2MPipelines.cpp:75-172`. The passes upstream of
`ttir-to-d2m` (multi-device annotation, `ttcore-register-device`,
`predicate-type-alignment`, `element-type-normalization`,
`decompose-composites`, `ttir-to-ttir-decomposition`, `explicate-tms`,
`erase-inverse-ops`, `move-reshape-to-constant`,
`fold-constant-reshape-broadcast`, `reduction-force-keep-dim`,
`rank-normalization`, `decompose-complex-reshape`, `implicit-broadcast-fold`,
canonicalizer, `global-data-format-conversion` (optional),
`decompose-complex-permute`) prepare TTIR for D2M ingestion and are out of
scope for this doc.

### 2.1 `ttir-to-d2m` (the D2M entry pass)

**Implementation**: `lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`. Pass options
(`TTIRToD2MOptions` in `include/ttmlir/Conversion/Passes.td`):
`default-input-memspace`, `default-output-memspace`, `ttnn-mode`,
`collapse-tensors-to-2d`, `enable-multicast-inference`.

**Input IR**: TTIR named ops on tensors. Reductions are guaranteed
`keep_dim=true` by the upstream prelude.

**Output IR**: Every compute op becomes a `d2m.generic` on tensors carrying a
`ttcore.metal_layout` encoding. Each `d2m.generic` has explicit `grid`,
`indexing_maps`, `iterator_types`, `threads`, and `block_factors`. Inside the
region, tile-level work is expressed via `linalg.generic` and D2M tile ops
(`d2m.tile_add`, `d2m.tile_matmul`, `d2m.tile_matmul_block`, etc.). Host/device
transitions appear as `d2m.to_layout` / `d2m.to_device` / `d2m.to_host`.

**Invariants on exit**:
- Tensors carry a `metal_layout` encoding.
- Every compute op is in a `d2m.generic`; no bare TTIR compute remains.
- The `ttnn-mode` flag has been propagated where it changes tensor
  translation (see §6 on the TTNN-mode variant).

### 2.2 `d2m-scalarize-const-tensors`

Splat constant fills are replaced by scalar fills inside generics.

### 2.3 `d2m-materialize-view-returns`

Any `d2m.generic` whose result is a view (no real data movement) is
materialized to ensure SSA invariants hold across consumers.

### 2.4 `d2m-grid-selection`

Chooses concrete grid shapes for each `d2m.generic` based on the device shape
and the op's iteration space. Honours `override-device-shape` and `ttnn-mode`.

### 2.5 `d2m-optimize-masks`

Optimizes masking patterns produced during conversion (e.g. constant masks
folded into iterator bounds) before they get decomposed.

### 2.6 `d2m-lower-to-layout`

Lowers high-level layout transformations (e.g. tilization, reblocking,
sharding rearrangement) into `d2m.to_layout` ops with explicit view layouts.

Followed by another `d2m-materialize-view-returns` to cover views introduced
by `lower-to-layout`.

### 2.7 `d2m-generic-fusion` (optional)

Driven by `enable-elementwise-fusion` and `enable-eltwise-reduction-fusion`.
Fuses adjacent compatible `d2m.generic` ops into a single fused generic; the
eltwise-reduction variant fuses an elementwise producer into a reduction
consumer with a single reduction dim.

### 2.8 Bufferization (the tensor → memref boundary)

`createTTIRBufferizationPipeline` is invoked here. It runs:

- `ttcore::createTTCoreOneShotBufferizePass()` by default — MetalLayoutAttr-aware,
  preserves device layouts.
- Standard MLIR `bufferization::createOneShotBufferizePass` with identity
  layout maps when `ttnn-mode=true` (lines 44-57).

**After bufferization**: tensors are gone. Region operands are memrefs with
TTCore layout encodings (`ShardLayoutAttr`, `CBLayoutAttr`, `InterleavedLayoutAttr`,
etc.) and a `memory_space` attribute (`DeviceL1` or `DeviceDRAM`). Inside
`d2m.generic` regions, operands are wrapped in `!d2m.cb<…>` (circular buffer)
types.

### 2.9 `d2m-insert-scratch-buffers`

Adds scratch buffers needed by fused generics for intermediate values.

### 2.10 `d2m-generic-apply-interchange`

Reorders the loop iterators of `d2m.generic` ops. Honours the
`matmul-interchange` option for matmul-shaped indexing maps.

### 2.11 `d2m-generate-outer-loops`

After this pass, every `d2m.generic` is in **Affine Blocked form**: the outer
iteration space (across grid blocks) is explicit as affine loops; the inner
work (per-tile or per-tile-block) is the generic body.

### 2.12 `d2m-decompose-masking`

Decomposes complex masking idioms into primitive `d2m.mask` patterns. Takes
`num-stream-buffers` as an option.

### 2.13 `d2m-allocate`

Memory planning. Assigns physical addresses and alignments to every memref
allocation. Resolves whether each generic operand needs a stream/CB or can
be aliased. After this pass:

- Every `memref.alloc()` carries `{address, alignment}` integer attributes.
- CB allocations are tagged with `#ttcore.cb_layout<capacity, num_buffers>`.
- L1/DRAM addresses honour `available-l1-addr-range` and stream insertion
  policy (`always` vs `infer`, controlled by `stream-insert-policy`).

### 2.14 `d2m-lower-multicast-loads`

Converts loads that broadcast across cores into multicast DMA forms.

### 2.15 `d2m-lower-to-explicit-form`

After this pass, every `d2m.generic` is in **Explicit Datamovement form**: the
region has explicit datamovement ops separated from compute, and remote
loads/stores live as `d2m.remote_load`/`d2m.remote_store` (or `d2m.local_copy`
for L1-to-L1).

### Frontend output contract

- Tensors are gone. All region operands are memrefs (often wrapped in
  `!d2m.cb`) with layout encodings and explicit `{address, alignment}`.
- Every `d2m.generic` is in Explicit Datamovement form with concrete grid,
  block factors, indexing maps, iterator types, and thread assignments.
- Layout transitions are explicit ops; no implicit view collapsing remains.
- Loops outside generics are affine and represent the grid-block iteration
  space.

## 3. The Backend Pipeline

`createD2MBackendPipeline` lives at `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp:174-259`.

### 3.1 `d2m-decompose-arange`

Decomposes `d2m.arange` into primitive index arithmetic ops.

### 3.2 `d2m-generic-tile-compute-loops`

Tiles compute loops inside generics so each tile-step fits the DST
register file. Honours `max-dst-physical-size-tiles`.

### 3.3 `d2m-linalg-to-affine`

Converts the `linalg.generic`-style compute regions inside `d2m.generic` into
affine loops. Marks root loops via `mark-root-loops=true`.

### 3.4 `d2m-op-scheduler`

Reorders compute ops within a generic to minimize DST usage. (Currently
unconditionally enabled; see `enable-op-scheduler` in `D2MPipelineOptions`.)

### 3.5 `d2m-insert-spill-and-scratch`

Inserts spill/refill code for intermediates that don't fit in DST.

### 3.6 `d2m-lower-scratch-allocate`

Lowers scratch allocations to concrete memrefs.

### 3.7 `d2m-insert-dst-register-access-unscheduled` / `-scheduled`

Two related passes that wrap compute ops in the DST register access protocol
(`d2m.acquire_dst`, copy-to/copy-from-dst nests). The unscheduled variant
operates before op-scheduler decisions are made for legacy paths; the
scheduled variant honours the schedule produced earlier. Both take
`max-dst-physical-size-tiles` and `enable-l1-acc`.

### 3.8 `d2m-insert-tile-matmul-block`

Replaces tile_matmul loop nests with `d2m.tile_matmul_block` calls when
beneficial. Driven by `use-tile-matmul`.

### 3.9 `d2m-sfpu-tile-loop-fission`

Splits SFPU compute into separate load / compute / store loops for better
DST reuse.

### 3.10 Affine flattening

`affine::createAffineLoopInvariantCodeMotionPass` (nested in `func.func`) →
`lower-affine` → `fold-memref-alias-ops` → `lower-affine` →
`d2m-generic-linearize-memref` → `lower-affine` again (to clean up
`affine.apply` ops generated by linearize).

After this stage compute ops live inside plain `scf.for`/`scf.if` loops with
flat memref indexing.

### 3.11 DMA lowering — frontend (compute/DM thread split)

- `d2m-hoist-cb-allocs` — Hoists CB allocations from inside `d2m.generic`
  bodies out to the operand position so they are shared across threads.
- `d2m-split-unified-thread` — Splits each `d2m.generic`'s unified region into
  separate compute and datamovement regions that share CBs.

At this point compute and DM threads exist as separate regions but the
explicit CB push/pop and DMA primitives are not yet in their final form.

### 3.12 DMA lowering — backend (concretize ops)

- `d2m-preallocate-mcast-semaphores` — Materializes
  `d2m.create_local_semaphore` / `d2m.create_global_semaphore` with stable IDs
  for multicast paths.
- `d2m-schedule-dma` — Distributes DMA work across threads, using
  pre-allocated semaphores to resolve producer/consumer dependencies.
- `d2m-lower-load-store-ops-to-dma` — Lowers `d2m.remote_load` /
  `d2m.remote_store` / `d2m.local_copy` into concrete `d2m.dma_read` /
  `d2m.dma_write` / `d2m.dma_wait` ops with explicit CB protocol
  (`d2m.reserve`, `d2m.push`, `d2m.wait`, `d2m.pop`).
- `d2m-optimize-dma` — Coalesces barriers, defers write barriers, etc.
- `d2m-expand-dma-read-composite-view` — Expands composite-view reads into
  primitive DMA ops.
- `d2m-lower-dma-to-fully-indexed-form` — Every DMA op becomes fully indexed
  with concrete source/destination address expressions.

### 3.13 `d2m-normalize-thread-args`

Inserts `d2m.get_arg` ops for any remaining additional thread arguments and
sets `resolution_stage` (`compile` or `runtime`) on `d2m.get_cb` /
`d2m.get_arg` so the D2M-to-TTKernel lowering can treat all arguments
uniformly.

### 3.14 Optimization sweep

Canonicalize, LICM, SCCP, CSE, arith int-range opts, LICM again.

### 3.15 `d2m-generic-regions-to-funcs`

Hoists each `d2m.generic` thread region out into a top-level
`func.func private @kernel_name` with a `d2m.thread = #d2m.thread<…>`
attribute. CB references inside the function are reached via `d2m.get_cb`;
thread arguments via `d2m.get_arg`.

### Backend output contract

- Each kernel thread is a top-level `func.func` tagged with a
  `d2m.thread` attribute (`#d2m.thread<compute>` /
  `#d2m.thread<datamovement>` / `#d2m.thread<unified>`) and an optional
  `kernelSymbol` / `nocIndex`.
- CB synchronization is explicit (`d2m.wait` / `d2m.reserve` / `d2m.push` /
  `d2m.pop`).
- DMA ops are fully indexed and barriers are normalized.
- Semaphores (`!d2m.local_semaphore`, `!d2m.global_semaphore`) carry stable
  IDs.

## 4. D2M → TTKernel / TTMetal / TTNN (the dispatch boundary)

The end-to-end pipeline `createTTIRToTTMetalPipeline`
(`lib/Dialect/D2M/Pipelines/D2MPipelines.cpp:310-354`) wraps each function in
a `ttcore.DeviceModuleOp`, hoists CPU-tagged ops out to a `CPUModuleOp`, then
runs the FE+BE inside the DeviceModule followed by these stages:

### 4.1 `addD2MToTTKernelPreEmitCPasses` (lines 283-291)

- `convert-d2m-to-ttkernel` (`lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`)
  — Replaces D2M tile ops with `ttkernel.*` builtins; `d2m.get_cb`-style
  references become `!ttkernel.cb` operands; CB sync ops
  (`d2m.wait/reserve/push/pop`) become
  `ttkernel.cb_wait_front/reserve_back/push_back/pop_front`; DMAs become
  `ttkernel.noc_async_read/write` and their barriers; semaphores become
  `ttkernel.noc_semaphore_*`. Accepts the `ttnn-mode` option.
- Canonicalizer.
- `ttkernel-control-dst-section` — Lowers DST section management.
- Optimization sweep (canon, LICM, SCCP, CSE, int-range).

> The pipeline intentionally stops short of EmitC here so that
> `convert-d2m-to-ttmetal` / `convert-d2m-to-ttnn` can still inspect TTKernel
> ops (e.g. `TypecastTileOp` locality for BFP8 unpack-mode selection).

### 4.2 Dispatch-level conversion

- `--ttnn-mode=false` → `createD2MToTTMetalPipeline` adds
  `convert-d2m-to-ttmetal` (`ConvertD2MToTTMetalOptions.mathFidelity =
  options.mathFidelity`). This wraps the per-thread kernel functions into a
  `ttmetal.enqueue_program` op carrying `ComputeConfigAttr` / `NocConfigAttr`
  / `EthernetConfigAttr` describing the dispatch profile.
- `--ttnn-mode=true` → `createD2MToTTNNPipeline` adds `convert-d2m-to-ttnn`
  (`lib/Conversion/D2MToTTNN/D2MToTTNN.cpp`). Each `d2m.generic` becomes a
  `ttnn.generic` packaging compute and data-movement kernels as
  `ttnn.compute_kernel` / `ttnn.data_movement_kernel` attributes with
  `ct_args` (compile-time) and `common_rt_args` (runtime). Memref boundary
  values are converted to `RankedTensorType` carrying `TTNNLayoutAttr`
  (scalar element type, not tile). See §6.

### 4.3 TTKernel finalization

- `ttkernel-hoist-inits` — Hoists TTKernel init ops out of inner loops to
  the function preamble.
- `ttkernel-insert-device-zone-scopes` (optional, gated on
  `insert-profiler-traces`) — Inserts `DeviceZone` scopes around selected
  TTKernel ops for the profiler.

### 4.4 EmitC lowering

- `convert-ttkernel-to-emitc` — Lowers TTKernel ops to EmitC for codegen.
- Canonicalizer.
- `emitc-form-expressions` — Hoists EmitC subexpressions.

The IR is now ready for serialization to either a TTMetal flatbuffer (default)
or a TTNN flatbuffer (`ttnn-mode=true`).

## 5. Memory & Memref Type Reference

This section is a compact reference to the types and attributes that appear
on memrefs and tensors during the pipeline. Definitions live primarily in
`include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td` (TTCore) and
`include/ttmlir/Dialect/D2M/IR/D2MOpsTypes.td` (D2M).

### 5.1 Memory spaces

`TTCore_MemorySpace` enum (`include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td`,
attached as the `memory_space` attribute on most memrefs):

| Value | Mnemonic | Meaning |
|---|---|---|
| `System` | `system` | Host CPU memory |
| `SystemMMIO` | `mmio` | Host MMIO mapped region |
| `DeviceDRAM` | `dram` | Device DRAM |
| `DeviceL1` | `l1` | Device L1 (core-local SRAM) |
| `RegisterDst` | `dst` | Device destination register file |

### 5.2 TTCore tensor / memref layout encodings

All defined in `TTCoreOpsTypes.td`. Each implements either
`TTCore_DeviceLayoutInterface` (tensor encodings) or `MemRefLayoutAttrInterface`
(memref encodings), or both.

| Attr | Mnemonic | Defined | Role | Introduced/used by |
|---|---|---|---|---|
| `MetalLayoutAttr` | `metal_layout` | TTCoreOpsTypes.td:382 | Tensor-level layout: logical_shape, dim_alignments, collapsed_intervals, memory_space, memory_layout. Encodes how a logical tensor maps to a physical sharded/tiled representation 1:1 with a memref | TTIR → D2M boundary; consumed by `ttir-to-d2m` and bufferization |
| `ViewLayoutAttr` | `view` | TTCoreOpsTypes.td:243 | View/reblock placeholder on a memref; the affine map on the producing `view_layout` op describes the actual rearrangement | `d2m-lower-to-layout`, view materialization |
| `ShardLayoutAttr` | `shard` | TTCoreOpsTypes.td:269 | Memref layout for a per-shard view distributed across a grid; carries per-dim byte stride and back-buffer count | Post-bufferization, on sharded memrefs |
| `CBLayoutAttr` | `cb_layout` | TTCoreOpsTypes.td:296 | Per-core L1 circular-buffer layout (always 1×1 grid, distinct from `shard`); carries stride + buffer count | Introduced by `d2m-allocate` and `d2m-hoist-cb-allocs` |
| `InterleavedLayoutAttr` | `interleaved` | TTCoreOpsTypes.td:322 | DRAM layout interleaved across all channels | Post-bufferization, on interleaved DRAM memrefs |
| `HostLayoutAttr` | `host_layout` | TTCoreOpsTypes.td:343 | Host-side padded memref layout; logical_shape, host_strides, host_volume, optional mesh | At host/device boundary |

### 5.3 Tile and DataType

| Type / Enum | Definition | Notes |
|---|---|---|
| `TTCore_TileType` | `TTCoreOpsTypes.td:709` | The hardware tile, default 32×32, parameterized by element `DataType`. Implements `MemRefElementTypeInterface` and `FloatTypeInterface` so memrefs can use it directly. Helper methods include `getSizeBytes`, `getHeight`, `getWidth`, `getElementType` |
| `TTCore_DataType` enum | `TTCoreOpsEnums.td:10-46` | `Float32`/`f32`, `Float16`/`f16`, `BFloat16`/`bf16`, `BFP_Float8`/`bfp_f8`, `BFP_BFloat8`/`bfp_bf8`, `BFP_Float4`/`bfp_f4`, `BFP_BFloat4`/`bfp_bf4`, `BFP_Float2`/`bfp_f2`, `BFP_BFloat2`/`bfp_bf2`, `UInt32`/`u32`, `UInt16`/`u16`, `UInt8`/`u8`, `Int32`/`si32`, plus a few extras |
| `TTCore_TensorMemoryLayout` enum | `TTCoreOpsEnums.td:206` | `Interleaved`, `Sharded` |
| `TTCore_IteratorType` enum | `TTCoreOpsEnums.td:82` | Used on `d2m.generic` `iterator_types`: `Parallel`, `Reduction` |
| `TTCore_GridAttr` | `TTCoreOpsTypes.td:30` | Grid shape with optional virt-to-physical affine map |

### 5.4 D2M types

Defined in `include/ttmlir/Dialect/D2M/IR/D2MOpsTypes.td`:

| Type | Mnemonic | Line | Role |
|---|---|---|---|
| `D2M_MemTx` | `mem_tx` | 20 | Memory-transaction handle returned by DMA ops; parameterized by `DMAType` so barriers can distinguish read/write/mcast-write |
| `D2M_LocalSemaphore` | `local_semaphore` | 32 | Core-local synchronization primitive |
| `D2M_CB` | `cb` | 37 | Circular-buffer wrapper around a memref (or, pre-bufferization, a tensor); implements `ShapedTypeInterface` and the Bufferization tensor/buffer-like interfaces |
| `D2M_GlobalSemaphore` | `global_semaphore` | 113 | Device-wide synchronization primitive |

### 5.5 D2M attributes and enums

`include/ttmlir/Dialect/D2M/IR/D2MOpsAttrs.td`:

| Attr | Mnemonic | Fields | Role |
|---|---|---|---|
| `D2M_ThreadAttr` | `thread` | `threadType` (enum), optional `kernelSymbol`, optional `nocIndex` | Tags a kernel function (or a `d2m.generic` region) with its thread role and NoC assignment |

`include/ttmlir/Dialect/D2M/IR/D2MOpsEnums.td`:

| Enum | Values | Purpose |
|---|---|---|
| `D2M_ReduceDim` | `R`, `C`, `RC` | Reduction dimension for tile-level reduce ops |
| `D2M_TileBcastType` | `None`, `Col`, `Row`, `Scalar` | Tile broadcast shape |
| `D2M_ThreadType` | `Compute`, `Datamovement`, `Unified` | Kernel thread role |
| `D2M_DMAType` | `Read`, `Write`, `McastWrite` | DMA transaction class (parameterizes `!d2m.mem_tx`) |
| `D2M_ResolutionStage` | `Compile`, `Runtime` | Tags whether a `d2m.get_cb` / `d2m.get_arg` operand is resolved at compile time or runtime |

### 5.6 `d2m.generic` attribute schema

`include/ttmlir/Dialect/D2M/IR/D2MOps.td:261`. Top-level region attributes:

| Attr | Source type | Meaning |
|---|---|---|
| `grid` | `TTCore_GridAttr` | Compute grid shape |
| `block_factors` | `I64ArrayAttr` | Per-iterator blocking factor (number of inner tiles per outer step) |
| `indexing_maps` | `AffineMapArrayAttr` | One affine map per operand mapping the iteration space to operand coordinates |
| `iterator_types` | `IteratorTypeArrayAttr` | Per-iterator parallel/reduction tag |
| `threads` | `D2M_ThreadArrayAttr` | One `ThreadAttr` per region (compute, datamovement, unified) |
| `fabricConnectionConfig` | optional `FabricConnectionConfigAttr` | Multicast/fabric topology for CCL paths |

### 5.7 TTKernel / TTMetal types (quick reference)

TTKernel types (`include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td`):

| Type | Role |
|---|---|
| `TTKernel_CB` (`cb`) | Circular buffer wrapping element count + element type (built from a memref post-bufferization) |
| `TTKernel_LocalSemaphore` (`local_semaphore`) | Core-local sync |
| `TTKernel_NocAddr` (`noc_addr`) | NoC address reference |
| `TTKernel_L1Addr`, `TTKernel_L1AddrPtr` | L1 address/pointer |
| `TTKernel_InterleavedAddrGenFast` (`interleaved_addr_gen_fast`) | Fast address generator for interleaved DRAM |
| `TTKernel_TensorAccessor` (`TensorAccessor`) | Encapsulates tensor access logic for a kernel argument |
| `TTKernel_FabricConnectionManager` (`FabricConnectionManager`) | Holds fabric connection state |

TTMetal types (`include/ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.td`):

| Type | Role |
|---|---|
| `TTMetal_LocalSemaphore` (`local_semaphore`) | Core-local sync |
| `TTMetal_GlobalSemaphore` (`global_semaphore`) | Device-wide sync |

TTMetal kernel configs (`TTMetalOpsAttrs.td`):

| Attr | Carries |
|---|---|
| `ComputeConfigAttr` | Kernel symbol, core range, args, math fidelity, unpack-to-dest modes |
| `NocConfigAttr` | Kernel, core range, args, NoC index |
| `EthernetConfigAttr` | Kernel, core range, args, ethernet type, NoC index |

## 6. Synchronization Lowering (top-down)

This section traces how synchronization is expressed at each level of the
pipeline, from the abstract `SynchronizableOpInterface` down to TTKernel CB
and semaphore primitives. The codebase has no separate `DFB` type; the
"data-flow buffer" role is filled entirely by CBs (`!d2m.cb` /
`!ttkernel.cb`) plus DMA-transaction barriers and semaphores.

### 6.1 The abstract layer: `SynchronizableOpInterface`

Definition: `include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.td:11-37`.

The interface exposes two methods:

```cpp
bool isProducer(mlir::OpOperand &operand);
bool isConsumer(mlir::OpOperand &operand);
```

A "producer" operand is one written to; a "consumer" operand is one read
from. Operations implementing this interface include the shard-level DMA
ops `d2m.remote_load`, `d2m.remote_store`, and `d2m.local_copy`
(`include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td`), plus the concrete
DMA ops that succeed them.

A related interface `D2M_ShardDMAOpInterface` (same file, line 55-71)
provides a uniform `getCBPort()` accessor for the shard-level DMA ops, so
scheduling and thread-split passes do not need to enumerate individual op
types.

These interfaces are the lever the pipeline uses to decide *where* CB
push/pop and semaphore inc/wait must eventually be inserted — **before**
any explicit sync primitives are materialized.

### 6.2 What's implicit, what's explicit (stage map)

| Stage | Synchronization form | Introduced/transformed by |
|---|---|---|
| TTIR | SSA data-dependence on tensors | n/a |
| Post-`ttir-to-d2m` | Region operands track producer/consumer roles; `SynchronizableOpInterface` tags ops; no explicit sync | `ttir-to-d2m` |
| Post-bufferization | `!d2m.cb` wraps memref operands inside regions; still no explicit CB ops | bufferization |
| Post-`d2m-hoist-cb-allocs` | CB allocations live outside generics; operands of generics are CB-typed | `d2m-hoist-cb-allocs` |
| Post-`d2m-split-unified-thread` | Region split into compute + datamovement; CB ops still not yet materialized | `d2m-split-unified-thread` |
| Mcast semaphore pre-allocation | `d2m.create_local_semaphore` / `d2m.create_global_semaphore` materialized with stable IDs | `d2m-preallocate-mcast-semaphores` |
| DMA scheduling | Producer/consumer dependencies resolved into ordered DMA ops across threads; uses pre-allocated semaphores | `d2m-schedule-dma` |
| Load/store → DMA lowering | `remote_load/store` / `local_copy` → `d2m.dma_read`/`d2m.dma_write` with **explicit CB protocol** (`d2m.reserve`, `d2m.push`, `d2m.wait`, `d2m.pop`) and `d2m.dma_wait` barriers | `d2m-lower-load-store-ops-to-dma` |
| DMA optimization | Barrier coalescing, write-barrier deferral; CB ops preserved | `d2m-optimize-dma` |
| Fully indexed DMA | Each DMA fully indexed; CB ops unchanged | `d2m-lower-dma-to-fully-indexed-form` |
| Backend conversion | `d2m.wait/reserve/push/pop` → `ttkernel.cb_wait_front/reserve_back/push_back/pop_front`; semaphore ops → `ttkernel.noc_semaphore_*` | `convert-d2m-to-ttkernel` |

### 6.3 Primitive vocabulary at each level

**D2M region-level CB protocol** (`include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td`):

| Op | Line | Role |
|---|---|---|
| `d2m.reserve` | 2583 | Producer reserves CB space |
| `d2m.push`    | 2607 | Producer commits / signals data ready |
| `d2m.wait`    | 2557 | Consumer blocks until data ready |
| `d2m.pop`     | 2647 | Consumer releases space |
| `d2m.dma_wait`| 1563 | Barrier on a `!d2m.mem_tx` handle |

**D2M semaphores** (`include/ttmlir/Dialect/D2M/IR/D2MOps.td` and
`D2MGenericRegionOps.td`):

| Op | Role |
|---|---|
| `d2m.create_local_semaphore` (D2MOps.td:563) | Allocate a local semaphore |
| `d2m.create_global_semaphore` (D2MOps.td:533) | Allocate a global (mcast) semaphore |
| `d2m.reset_global_semaphore` (D2MOps.td:550) | Reset semaphore value |
| `d2m.semaphore_set`, `d2m.semaphore_inc`, `d2m.semaphore_wait` (D2MGenericRegionOps.td:2309-2313) | Region-level semaphore ops |

**TTKernel CB primitives** (`include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td`):

| Op | Line | Role |
|---|---|---|
| `ttkernel.cb_push_back`    | 2874 | Producer pushes pages |
| `ttkernel.cb_pop_front`    | 2889 | Consumer pops pages |
| `ttkernel.cb_reserve_back` | 2904 | Producer reserves pages |
| `ttkernel.cb_wait_front`   | 2919 | Consumer waits for pages |

**TTKernel semaphore / NoC barrier primitives** (same file):

| Op | Line | Role |
|---|---|---|
| `ttkernel.get_semaphore` | 3500 | Get a semaphore handle |
| `ttkernel.noc_semaphore_inc` | 3514 | Inc a semaphore via NoC |
| `ttkernel.noc_semaphore_set` | 3551 | Set a semaphore via NoC |
| `ttkernel.experimental::semaphore_wait` | 3567 | Wait on a semaphore |
| `ttkernel.experimental::semaphore_wait_min` | 3583 | Wait on a min value |
| `ttkernel.noc_semaphore_set_multicast` (+ loopback variant) | 3599 / 3628 | Multicast semaphore set |
| `ttkernel.noc_semaphore_inc_multicast` | 3656 | Multicast semaphore inc |
| `ttkernel.noc_async_read_barrier`, `ttkernel.noc_async_write_barrier` | (search file) | DMA barriers |

### 6.4 Cheat sheet (intent → primitive)

| Intent | Region-level (D2M) | Final (TTKernel) |
|---|---|---|
| Producer reserves L1 space for a tile | `d2m.reserve` | `ttkernel.cb_reserve_back` |
| Producer signals data ready | `d2m.push` | `ttkernel.cb_push_back` |
| Consumer waits for data | `d2m.wait` | `ttkernel.cb_wait_front` |
| Consumer releases space | `d2m.pop` | `ttkernel.cb_pop_front` |
| Wait on a specific DMA transaction | `d2m.dma_wait` on `!d2m.mem_tx` | `ttkernel.noc_async_*_barrier` |
| Cross-core mcast handshake | `d2m.create_global_semaphore` + `semaphore_set/inc/wait` | `ttkernel.noc_semaphore_*` |

### 6.5 Reference test files

- `test/ttmlir/Conversion/D2MToTTKernel/cb_port_custom_ports.mlir` — CB sync
  ops in compute functions after thread split.
- `test/ttmlir/Conversion/D2MToTTKernel/semaphores.mlir` — semaphore lowering.
- `test/ttmlir/Conversion/D2MToTTMetal/generic_lowering.mlir` — full chain
  through to `ttmetal.enqueue_program`.

## 7. The TTNN-Mode Variant (two invocation paths)

`ttnn-mode` is *not* a ttnn-jit-only flag. It is the `ttnn-mode` option on
`D2MPipelineOptions`
(`include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h:175-177`, default
`false`) and is used by **two** distinct callers, both of which produce TTNN
IR (`ttnn.generic`) instead of TTMetal IR. The differences live above and
below the D2M pipeline, not inside it.

### 7.1 Two invocation paths

| Path | Caller | Scope | Allocation responsibility |
|---|---|---|---|
| A. ttnn-jit | Python `@ttnn_jit.jit()` decorator | Whole user function | Host Python |
| B. TTNN-pipeline D2M-subgraph carve-out | `createTTNNCreateD2MSubgraphs` inside `createTTIRToTTNNCommonPipeline` | Per-subgraph (eltwise chains wrapped in `ttnn.d2m_subgraph`) | TTNN runtime |

Both paths set `ttmetalOptions.ttnnMode = true` and reuse the same
`D2MFrontendPipeline` / `D2MBackendPipeline` / `D2MToTTKernelPipeline`.

### 7.2 What the flag changes inside D2M

| Stage | `ttnn-mode=false` | `ttnn-mode=true` |
|---|---|---|
| `ttir-to-d2m` | standard tensorization | takes the `ttnnMode` option; affects grid selection and TTNN tensor translation |
| Bufferization | `ttcore::createTTCoreOneShotBufferizePass` (MetalLayoutAttr-aware) | standard MLIR `createOneShotBufferizePass` with identity layout maps (D2MPipelines.cpp:44-57) |
| Dispatch-level final conversion | `convert-d2m-to-ttmetal` | `convert-d2m-to-ttnn` |
| Output IR | `ttmetal.enqueue_program` + ComputeConfig/NocConfig/EthernetConfig | `ttnn.generic` + `ttnn.compute_kernel` / `ttnn.data_movement_kernel` |
| Boundary tensor type | memref with `MetalLayoutAttr` / sharded encoding | `RankedTensorType` with `TTNNLayoutAttr` (scalar element type, not tile) |
| Flatbuffer producer | TTMetal flatbuffer | TTNN flatbuffer (`ttnnToFlatbuffer`) |

### 7.3 Path A: ttnn-jit (Python decorator)

- Trigger: `@ttnn_jit.jit(...)` decorator. CLI form:
  `--ttir-to-ttmetal-pipeline="ttnn-mode=true set-math-fidelity=HiFi4"`.
- Build flag: `-DTTMLIR_ENABLE_TTNN_JIT=ON`.
- Compiles an entire user function through TTIR → D2M → TTNN.
- JIT caching: `ProgramDescCache` nested in the binary object.
- Constraint envelope, supported ops, datatypes and fusion limits live in
  the [ttnn-jit docs](./ttnn-jit.md). The constraints come from the TTNN
  side; they apply to *what the user can write*, not to what D2M does
  internally.

### 7.4 Path B: D2M-subgraph carve-out from the main TTNN pipeline

The main TTIR→TTNN pipeline can opportunistically carve elementwise chains
out of a TTNN program, compile them through the D2M pipeline (in
ttnn-mode), and re-inline the compiled `ttnn.generic` back into the host
TTNN function.

**Enabling options** (`include/ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h`):

- `enableCreateD2MSubgraphs` — turn on the carve-out. Requires the TTNN
  optimizer (auto-enabled with a warning if not already on).
- `enableD2MElementwiseFusion` — auto-enabled by
  `resolveCreateD2MSubgraphsOptions()` when subgraphs are enabled; feeds
  `d2m-generic-fusion` inside the D2M FE.

**Pass sequence in the TTNN pipeline**
(`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`):

1. `--ttnn-create-d2m-subgraphs` (TTNNPipelines.cpp:363-371) — identifies
   maximal elementwise chains (unary + binary), outlines each into a private
   function, and wraps it in a `ttnn.d2m_subgraph` op carrying a symbol
   reference. Implementation:
   `lib/Dialect/TTNN/Transforms/TTNNCreateD2MSubgraphs.cpp`.
2. `createTTNNPipelineD2MPass` (TTNNPipelines.cpp:386-390; impl at
   614-628) — runs `convert-ttnn-to-ttir` first to translate the TTNN
   subgraph back into TTIR, then invokes the **full**
   `D2MFrontendPipeline` + `D2MBackendPipeline` + `D2MToTTKernelPipeline` +
   `D2MToTTNNPipeline` with `ttmetalOptions.ttnnMode = true`. After this
   pass the subgraph function body holds `ttnn.generic` ops + kernel
   functions.
3. `--ttnn-collaspe-d2m` (TTNNPipelines.cpp:390) — inlines the compiled
   `ttnn.d2m_subgraph` body back into the parent function and deletes the
   private symbol.

**Op anchor**: `TTNN_D2MSubgraphOp`
(`include/ttmlir/Dialect/TTNN/IR/TTNNOps.td:3539`) — references a private
function holding the subgraph; pre-compile it holds TTNN ops, post-compile
it holds `ttnn.generic` + kernel funcs. Verify and helpers in
`lib/Dialect/TTNN/IR/TTNNOps.cpp:5802-5870`.

**Standalone pipeline registration**: `--ttnn-through-d2m-pipeline`
(TTNNPipelines.cpp:760-767) exposes the carve-out + D2M-compile step on its
own.

**Reference tests**:

- `test/ttmlir/Dialect/TTNN/dispatch_d2m/create_d2m_subgraphs.mlir`
- `test/ttmlir/Dialect/TTNN/dispatch_d2m/collaspe.mlir`
- `test/ttmlir/Dialect/TTNN/dispatch_d2m/ttnn_d2m_e2e.mlir` (end-to-end)
- `test/ttmlir/Dialect/TTNN/optimizer/d2m_optimizer_two_d2m_subgraphs.mlir`

### 7.5 `convert-d2m-to-ttnn` internals

`lib/Conversion/D2MToTTNN/D2MToTTNN.cpp`:

- `convertMemrefToTTNNTensor()` — D2M memref → TTNN tensor with
  `TTNNLayoutAttr`.
- `getTTNNLayoutFromDeviceLayout()` — maps ttcore device layout to TTNN
  `TensorMemoryLayout` (`Interleaved` / `BlockSharded` / `HeightSharded` /
  `WidthSharded`) and `BufferType` (L1/DRAM).
- `convertMathFidelity()` — maps TTMetal `MathFidelity` to TTNN
  `ComputeKernelMathFidelity`.
- Each `d2m.generic` becomes one `ttnn.generic` packaging compute and
  data-movement kernels with `ct_args` (compile-time) and `common_rt_args`
  (runtime).

### 7.6 Runtime integration

- Both paths produce a TTNN flatbuffer that is consumed by the TTNN runtime,
  not the TTMetal runtime.
- Path A caches via `ProgramDescCache` in the JIT binary.
- Path B's compiled `ttnn.generic` ops live inside the host TTNN program and
  execute as part of normal TTNN dispatch.

### 7.7 Cross-references

- [ttnn-jit](./ttnn-jit.md) for the Python-side usage and full Path-A
  constraint list.
- `include/ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h` for
  `enableCreateD2MSubgraphs` / `enableD2MElementwiseFusion`.

## 8. Reading the Pipeline in Practice

### 8.1 Dumping IR after each pass

```sh
ttmlir-opt --ttir-to-ttmetal-pipeline=… \
           --mlir-print-ir-after-all \
           --mlir-disable-threading \
           input.mlir 2>ir.log
```

To isolate the FE or BE:

```sh
ttmlir-opt --d2m-fe-pipeline=… input.mlir       # stops at end of FE
ttmlir-opt --d2m-be-pipeline=… input.afterFE.mlir
```

### 8.2 Canonical example tests by stage

| Stage | Where to look |
|---|---|
| TTIR → D2M | `test/ttmlir/Conversion/TTIRToD2M/` |
| Post-`d2m-allocate` | `test/ttmlir/Dialect/D2M/Allocate/` |
| D2M → TTKernel (tile + CB ops) | `test/ttmlir/Conversion/D2MToTTKernel/` |
| D2M → TTMetal (full chain) | `test/ttmlir/Conversion/D2MToTTMetal/` |
| D2M → TTNN (sanity + e2e) | `test/ttmlir/Conversion/D2MToTTNN/`, `test/ttmlir/Dialect/TTNN/dispatch_d2m/` |
| End-to-end on silicon | `test/ttmlir/Silicon/D2M/` |

## 9. Cross-References

- [Overview](./overview.md) — full dialect overview and historical D2M
  context.
- [ttnn-jit](./ttnn-jit.md) — Python-side ttnn-jit usage.
- `include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h` — pipeline options
  struct.
- `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` — pipeline assembly.
- `include/ttmlir/Dialect/D2M/Transforms/Passes.td` — pass declarations.
- `include/ttmlir/Dialect/D2M/IR/D2MOps.td`,
  `D2MGenericRegionOps.td`,
  `D2MOpsTypes.td`,
  `D2MOpsAttrs.td`,
  `D2MOpsEnums.td`,
  `D2MGenericRegionOpsInterfaces.td` — D2M op/type definitions and
  interfaces.
- `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td`,
  `TTCoreOpsEnums.td` — TTCore layout/memref attributes and enums.
- `include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td`,
  `TTKernelOpsTypes.td` — TTKernel ops and types.
- `include/ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.td`,
  `TTMetalOpsAttrs.td` — TTMetal kernel configs.
- `lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`,
  `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`,
  `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp`,
  `lib/Conversion/D2MToTTNN/D2MToTTNN.cpp` — conversion implementations.
- `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` — TTNN pipeline (D2M
  subgraph carve-out lives here).
