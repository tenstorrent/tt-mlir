# D2M DFB Codegen Bring-Up (Phase 1: SPSC + 1P→nC compute)

## Context

Quasar's Tensix Neo replaces the Wormhole/Blackhole **Circular Buffer (CB)** primitive with the **Dataflow Buffer (DFB)** — an MPMC FIFO backed by per-consumer hardware tile counters with explicit `posted`/`acked` pairs. CBs do not exist on Quasar; targeting Quasar requires the compiler to emit DFB host setup (`CreateDataflowBuffer`, `BindDataflowBufferToProducerConsumerKernels`) and DFB kernel ops (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`finish`) in place of CB ops.

D2M today emits only SPSC dataflow (1 DM producer → 1 compute consumer through one CB). The existing pipeline correctly handles the 1P→nC compute case as well — multiple compute threads share the same CB, with `mhartid` (surfaced as `d2m.my_thread_id`) differentiating them inside an identical kernel binary, and the current sync scaffolding (semaphores) coordinating fan-out.

**Key realization that shapes this design:** the Quasar `DataflowBuffer` kernel-side device API (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`finish`) takes **no consumer-slot argument**. Each consumer hart picks its tile-counter slot implicitly via `mhartid`, driven by host-side `BindDataflowBufferToProducerConsumerKernels`. The compute kernel binary is genuinely identical across the N consumer threads. Nothing on the kernel-emission side needs a per-thread compile-time slot value.

That fact collapses the design: **DFB-vs-CB is purely a lowering-time concern, not an IR-level concept the D2M middleend needs to reason about.** The D2M layer stays CB-native; the choice between emitting `ttkernel.cb_*` and `ttkernel.dfb_*` (and between `ttmetal.create_buffer` and `ttmetal.create_dataflow_buffer`) happens at the conversion boundary, gated by a single pipeline flag. CB tests on WH/BH continue to work unchanged.

**Scope (in this plan):**
- Ops: matmul (existing 4-compute-thread tiling: `enable-compute-thread-tiling=true num-compute-threads=4`), eltwise unary, eltwise binary.
- Topologies: 1P→1C (drop-in for CB), 1P→nC where nC = compute-thread count (BLOCKED on the consumer side, broadcast input).
- Sync mode: explicit (`reserve_back`/`push_back`/`wait_front`/`pop_front` + `finish`). No implicit transaction-ID sync.
- End-to-end: conversion-time swap + TTMetal/flatbuffer schema + runtime executor + Quasar silicon test.

**Out of scope:** nP→1C, nP→nC, STRIDED consumer patterns, implicit sync, reductions, the per-program 8-DFB-limit fallback. If future work needs the optimizer/scheduler to reason about cardinality at the D2M level (e.g., choosing nP→1 over 1P→1), the D2M layer can grow a DFB vocabulary then. For now, do not.

**Key constraint:** The N compute threads remain a single `scf.forall` (already in IR) lowered by `D2MMaterializeComputeThreadForall` to one SPMD compute kernel using `d2m.my_thread_id`. The compute kernel binary must be **identical across all N compute threads**. The existing CB pipeline already produces this shape; the conversion-time DFB swap inherits it. Cardinality (num_consumers = N) is plumbed only into the host-side `#ttmetal.dfb_config` — it does not appear in any kernel op.

## Approach: late-switch DFB lowering

Keep the D2M layer entirely CB-native. Add DFB ops at the TTKernel and TTMetal layers (kernel-side ABI + host-side ABI). Flip lowering targets at the conversion boundary based on a `use-dfbs` pipeline flag.

### A. IR design — TTKernel + TTMetal only

**TTKernel layer** (already landed in PR3 of the in-progress branch):
- `!ttkernel.dfb<num_entries, element_type, num_producers, num_consumers>` type.
- `ttkernel.dfb_reserve_back`, `dfb_push_back`, `dfb_wait_front`, `dfb_pop_front`, `dfb_finish` ops. Same shape as the CB ops; `dfb_wait_front` carries `TTKernel_DeviceZoneOpTrait` for parity. **None of these take a consumer-slot operand.**
- `ArgType::DFBId` for compile-time arg plumbing of the logical DFB id.

**TTMetal layer** (already landed in PR3 of the in-progress branch):
- `#ttmetal.dfb_config` carrying `entry_size`, `num_entries`, `num_producers`, `num_consumers`, `producer_risc_mask`, `consumer_risc_mask`, `producer_pattern`, `consumer_pattern`, `enable_implicit_sync`, `data_format`.
- `ttmetal.create_dataflow_buffer(core_range, config) -> ui32` — logical DFB id.
- `ttmetal.bind_dfb_to_kernels(dfb_id, producer_kernel: symbol, consumer_kernel: symbol)`.
- `ttmetal.enqueue_program` extended with `$dfbs` + `$dfb_ids` operand list parallel to `$cbs` + `$cb_ports`.

**Not needed (revised from earlier plan, work to be rolled back):**
- `#ttcore.dfb_layout` attribute. D2M memrefs retain `#ttcore.cb_layout`; cardinality lives only in `#ttmetal.dfb_config`, constructed at conversion time.
- `!d2m.dfb` type and the `d2m.get_dfb` / `d2m.dfb_wait` / `d2m.dfb_reserve` / `d2m.dfb_push` / `d2m.dfb_pop` / `d2m.dfb_finish` ops. The D2M layer stays CB-native.
- The `#ttcore.dfb_access_pattern` enum is retained because `#ttmetal.dfb_config` uses it.

### B. Pass changes

Pipeline reference: `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp`.

1. **All D2M middleend passes** — `D2MAllocate`, `D2MHoistCBAllocs`, `D2MMaterializeComputeThreadForall`, `D2MNormalizeThreadArgs`, etc.: **unchanged.** They continue to operate on CB layouts/types and CB sync ops. No flag plumbing. No new code.

2. **`ConvertD2MToTTKernel`** (`lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`). Add `useDFBs` pass option. When the flag is on:
   - Rewrite each `D2M*CB*Rewriter` (the existing CB-op→TTKernel-CB-op patterns at ~L2794 / ~L3251) to emit `ttkernel.dfb_*` ops instead of `ttkernel.cb_*`.
     - `d2m.wait` → `ttkernel.dfb_wait_front` (instead of `cb_wait_front`).
     - `d2m.reserve` → `ttkernel.dfb_reserve_back`.
     - `d2m.push` → `ttkernel.dfb_push_back`.
     - `d2m.pop` → `ttkernel.dfb_pop_front`.
   - `D2MGetCBRewriter` (~L2794–2821): when flag is on, append `ArgType::DFBId` to the function `ArgSpec` instead of `ArgType::CBPort`; produce a `!ttkernel.dfb` SSA value (via a small `unrealized_conversion_cast` or a similar adapter) for downstream sync ops to consume.
   - Insert `ttkernel.dfb_finish` at the end of every producer-kernel func body. Producer = a thread region that performs any `d2m.reserve` / `d2m.push` on a CB that becomes a DFB. Failing to emit `dfb_finish` silently hangs Quasar. Implementation: walk the function for ops that produce on each DFB-typed value and ensure exactly one `dfb_finish` exists on every exit path.

   The compute-thread case (1P→nC) needs no special handling here. Each compute thread runs the same kernel binary; each `ttkernel.dfb_wait_front` / `dfb_pop_front` is the unmodified consumer-side ABI; the slot assignment happens host-side at bind time.

3. **`ConvertD2MToTTMetal`** (`lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp`). Add `useDFBs` pass option. When the flag is on:
   - In `MemrefAllocRewriter` (~L295), branch on the CB's role/cardinality: emit `ttmetal.create_dataflow_buffer` (with `#ttmetal.dfb_config`) instead of `ttmetal.create_buffer`. Cardinality derivation:
     - `num_producers` = count of distinct producer thread regions in the parent `d2m.generic` that write this operand. MVP: always 1.
     - `num_consumers` = count of distinct consumer thread regions × `numComputeThreads` if the consumer is a compute thread under `enableComputeThreadTiling`; otherwise count of distinct consumer thread regions.
     - `producer_pattern` = STRIDED (MVP). `consumer_pattern` = BLOCKED when `num_consumers > 1` (broadcast input to N compute threads); STRIDED otherwise.
     - `producer_risc_mask` / `consumer_risc_mask` derived from the producer/consumer thread types (DM bits 0–7, compute bits 8–15).
   - Route DFB-flavored buffers into the new `$dfbs` / `$dfb_ids` operands on `ttmetal.enqueue_program` instead of `$cbs` / `$cb_ports`.
   - For each DFB created, emit a `ttmetal.bind_dfb_to_kernels` referencing the producer and consumer kernel func symbols (the kernels were outlined by `D2MGenericRegionsToFuncs` before this pass). For 1P→nC, the single compute kernel symbol is the consumer; the runtime allocates N TC slots from `num_consumers` in the config.

4. **`ConvertTTKernelToEmitC`** (`lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` ~L1362). Add opaque rewriters for the new TTKernel DFB ops:
   - Per-kernel-func prologue (once per DFB id): `experimental::DataflowBuffer dfb_<id>(static_cast<uint16_t>(<get_compile_time_arg_val>));`
   - `dfb.reserve_back(n);`, `dfb.push_back(n);`, `dfb.wait_front(n);`, `dfb.pop_front(n);`, `dfb.finish();`
   - Constructor form: `DataflowBuffer(uint16_t logical_dfb_id)`. The id flows through a single ct_args slot via the new `ArgType::DFBId`.

### C. Pipeline option

Add to `include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h::D2MPipelineOptions`:

```
Option<bool> useDFBs{*this, "use-dfbs",
    llvm::cl::desc("Lower CB sync ops to Quasar Dataflow Buffer ops "
                   "instead of Circular Buffer ops; required for Quasar."),
    llvm::cl::init(false)};
```

Plumbed into `ConvertD2MToTTKernelOptions.useDFBs` and `ConvertD2MToTTMetalOptions.useDFBs` only. No D2M-transform pass takes this flag. CLI: `--ttir-to-ttmetal-pipeline="use-dfbs=true ..."`. Default off. Replace with arch-driven defaulting once a `Quasar` `Arch` enum lands.

### D. Host-side / flatbuffer / runtime

Flatbuffer schema (`include/ttmlir/Target/TTMetal/`):
- `types.fbs`: `enum DFBAccessPattern : ushort { STRIDED, BLOCKED }`, `table DataflowBufferConfig {...}`, `table DFBRef { global_id, dfb_config, core_range_set }`.
- `command.fbs`: `table CreateDataflowBufferCommand { ref: DFBRef; }`, `table BindDFBToKernelsCommand { dfb_global_id, producer_kernel_symbol, consumer_kernel_symbol }`; add to `CommandType` union.
- `program.fbs`: `KernelArgDFBId { operand_idx: uint32 }`; add to `KernelArgType` union (parallel to `KernelArgCBPort`).
- `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp`: serialize the new ops and the new ArgType. (The stub `llvm_unreachable` for `DFBId` already added in PR3 is replaced with real serialization here.)

Runtime (`runtime/lib/ttmetal/`):
- `executor.cpp`: handle `CreateDataflowBufferCommand` → `experimental::dfb::CreateDataflowBuffer`; `BindDFBToKernelsCommand` → `experimental::dfb::BindDataflowBufferToProducerConsumerKernels`. Maintain a per-program DFB id table. Resolve producer/consumer `KernelHandle` from symbol via the program's kernel registry.
- `executor.h`, `executor_utils.h`, `arguments.h`, `kernels.{cpp,h}`: parallel additions for `KernelArgDFBId` ct_args plumbing.
- **Arch gating in runtime:** dispatching `CreateDataflowBufferCommand` against a non-Quasar device emits a runtime error `"DFB ops require Quasar"`.

### E. Tests

Lit (under `test/ttmlir/`):
- `Conversion/D2MToTTKernel/d2m_to_ttkernel_dfb.mlir` — `--ttir-to-ttkernel-pipeline=use-dfbs=true`: verify the same D2M input that produces `ttkernel.cb_*` without the flag produces `ttkernel.dfb_*` with the flag, including `ArgType::DFBId` in ArgSpec and `dfb_finish` at end of producer.
- `Conversion/D2MToTTMetal/d2m_to_ttmetal_dfb.mlir` — verify `ttmetal.create_dataflow_buffer` + `ttmetal.bind_dfb_to_kernels` for 1P→1C eltwise and 1P→4C matmul; verify `$dfbs` / `$dfb_ids` are populated and `$cbs` / `$cb_ports` are empty.
- `Conversion/TTKernelToEmitC/ttkernel_to_emitc_dfb.mlir` — verify emitted `DataflowBuffer dfb(id); dfb.reserve_back(...)` form.
- **Regression guard:** existing matmul/eltwise lit tests without `use-dfbs` remain bit-identical (no D2M passes touched).

Silicon end-to-end (under `test/ttmlir/Silicon/D2M/quasar/`):
- `eltwise_unary.mlir` — 1P→1C.
- `eltwise_binary.mlir` — 2× 1P→1C input + 1× 1C→1P output.
- `matmul_4ct.mlir` — `use-dfbs=true,enable-compute-thread-tiling=true,num-compute-threads=4`; 1P→4C input DFB for the broadcast operand.

### F. Suggested PR sequence (from here forward)

1. **Roll back over-built IR.** Remove `!d2m.dfb`, `d2m.get_dfb`, `d2m.dfb_*` region ops (PR2 work) and the `#ttcore.dfb_layout` attribute (PR1 portion). Keep `!ttkernel.dfb` / `ttkernel.dfb_*` / `ArgType::DFBId` (PR3) and `#ttmetal.dfb_config` / `ttmetal.create_dataflow_buffer` / `ttmetal.bind_dfb_to_kernels` / enqueue_program `$dfbs`/`$dfb_ids` (PR3). Keep also the `#ttcore.dfb_access_pattern` enum since `#ttmetal.dfb_config` uses it. Delete the now-stale parser round-trip tests for the dropped attrs/ops.

2. **`ConvertD2MToTTKernel` DFB patterns** — `useDFBs` option; CB→DFB pattern set. `D2MGetCBRewriter` flag-conditioned to emit `ArgType::DFBId`. `dfb_finish` insertion. (File: `D2MToTTKernel.cpp`.)

3. **`ConvertD2MToTTMetal` DFB lowering** — `useDFBs` option; emit `create_dataflow_buffer` + `bind_dfb_to_kernels`; cardinality inference from thread-region structure; route DFB buffers into `$dfbs`/`$dfb_ids`. (File: `D2MToTTMetal.cpp`.)

4. **`ConvertTTKernelToEmitC` DFB rewriters** — emit `DataflowBuffer` C++ calls. (File: `TTKernelToEmitC.cpp`.)

5. **Flatbuffer schema + runtime executor** — end-to-end silicon path; replace the `DFBId` `llvm_unreachable` with real serialization.

Each PR is independently buildable; PRs 2–4 are testable in isolation via FileCheck. End-to-end silicon does not light up until PR5.

Follow-up (out of this plan): add `Arch::Quasar` and switch `use-dfbs` to arch-driven default; nP→nC topologies (likely re-introduces D2M-level DFB IR for cardinality reasoning); implicit-sync mode; per-program 8-DFB-limit fallback.

## Critical files

**Modified going forward:**
- `include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h` (add `useDFBs` option)
- `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` (plumb option into two conversion passes)
- `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` (DFB sync patterns, `dfb_finish` insertion)
- `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` (DFB host-side lowering, cardinality inference)
- `lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` (DFB C++ emission)
- `include/ttmlir/Target/TTMetal/types.fbs`, `command.fbs`, `program.fbs`
- `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp`
- `runtime/lib/ttmetal/executor.{h,cpp}`, `executor_utils.h`, `arguments.h`, `kernels.{cpp,h}`

**Rolled back from PR1 and PR2:**
- `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td` — drop `TTCore_DFBLayoutAttr`. Keep `TTCore_DFBAccessPatternAttr` (still referenced by `#ttmetal.dfb_config`).
- `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td` — keep `TTCore_DFBAccessPattern` (still referenced by `#ttmetal.dfb_config`).
- `lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp` — drop `DFBLayoutAttr` static getters and `getAffineMap`.
- `include/ttmlir/Dialect/D2M/IR/D2MOpsTypes.td` — drop `D2M_DFB`.
- `include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td` — drop `D2M_DFBOp` class, `D2M_DFBWaitOp` / `DFBReserveOp` / `DFBPushOp` / `DFBPopOp` / `DFBFinishOp`, `D2M_GetDFBOp`.
- `lib/Dialect/D2M/IR/D2MGenericRegionOps.cpp` — drop the parallel `bufferizeDFBOp` helper and the `DFBPushOp` / `DFBPopOp` method definitions.
- `test/ttmlir/Dialect/TTCore/dfb_layout_attr.mlir`, `test/ttmlir/Dialect/D2M/dfb_type.mlir`, `test/ttmlir/Dialect/D2M/dfb_ops.mlir` — delete.

**Kept from PR1 and PR3:**
- `include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td` — `!ttkernel.dfb` type.
- `include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.td` — `ArgType::DFBId`.
- `include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` — `ttkernel.dfb_*` ops.
- `lib/Dialect/TTKernel/IR/TTKernelOps.cpp` — DFB verifier impls.
- `include/ttmlir/Dialect/TTMetal/IR/TTMetalOpsAttrs.td` — `#ttmetal.dfb_config`.
- `include/ttmlir/Dialect/TTMetal/IR/TTMetalOps.td` — `ttmetal.create_dataflow_buffer`, `ttmetal.bind_dfb_to_kernels`, `$dfbs`/`$dfb_ids` on `enqueue_program`.
- `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp` — the `DFBId` unreachable stub (replaced with real serialization in PR5 of the new sequence).
- `lib/Conversion/D2MToTTNN/D2MToTTNN.cpp` — the `DFBId` unreachable in the TTNN target.
- `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` — the empty `dfbs` / `dfb_ids` passed to existing `EnqueueProgramOp` builds.
- `test/ttmlir/Dialect/TTKernel/dfb_type.mlir`, `test/ttmlir/Dialect/TTKernel/dfb_ops.mlir`, `test/ttmlir/Dialect/TTMetal/dfb_config_attr.mlir`, `test/ttmlir/Dialect/TTMetal/dfb_host_ops.mlir` — keep.
- Updates to `test/ttmlir/Conversion/D2MToTTMetal/{generic_lowering,spatial_lowering}.mlir` and `test/python/golden/d2m/fabric_api_snippets/*.mlir` — keep (these reflect the new `enqueue_program` operand shape).

**Reused existing utilities:**
- CB allocation/hoist/normalize/conversion patterns in D2M — **unchanged**, serve as the input to the new conversion-time CB→DFB rewriters.
- `D2MGetCBRewriter` and the `D2MCBOpRewriter<...>` family in `D2MToTTKernel.cpp` are the templates for the flag-conditioned variants.
- Existing `MemrefAllocRewriter` in `D2MToTTMetal.cpp:295` is the model for branching to `create_dataflow_buffer`.
- `experimental::DataflowBuffer` device API at `tt-metal-main/tt_metal/hw/inc/experimental/dataflow_buffer.h`.
- Host API at `tt-metal-main/tt_metal/api/tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp`.
- Scoping doc `docs/src/specs/d2m-dataflow-buffer-bringup.md`. This plan diverges from the scoping doc's "Phase 2: Cardinality in the IR" — under late-switch, cardinality is not in the D2M IR; it's reconstructed at the conversion boundary from the existing thread-region structure.

## Risks / unknowns

1. **`risc_mask` derivation.** A single utility `deriveDfbRiscMask(KernelFuncOp, role)` walks the kernel func's thread type/id to produce the bitmask. Getting this wrong silently mis-binds the DFB. Unit-test the utility standalone.

2. **`dfb_finish` placement.** No CB analogue; failing to emit is a silent hang on Quasar. Implementation lives in `ConvertD2MToTTKernel`: after all producer-side patterns rewrite, walk the kernel func and ensure exactly one `dfb_finish` exists per DFB-typed value on every kernel-exit path. Add a verifier check that emits a clear error if missing.

3. **Cardinality inference at conversion time.** Under late-switch, the conversion pass has to reconstruct `num_producers` / `num_consumers` from the parent `d2m.generic`'s thread regions and the surrounding pipeline flags (`enableComputeThreadTiling`, `numComputeThreads`). The D2M IR encodes this via the thread region structure already, so it's derivable — but the inference logic is concentrated in one place rather than tied to a memref attribute. Add a verifier on `#ttmetal.dfb_config` that catches obvious inconsistencies (e.g., `num_consumers > 1` with STRIDED consumer pattern in MVP scope is a bug).

4. **TTMetal symbol resolution at runtime.** `bind_dfb_to_kernels` references kernels by symbol. The runtime resolves to `KernelHandle` via the program's kernel registry. Verify the registry is populated in dispatch order such that `BindDFBToKernelsCommand` can resolve symbols when it runs.

5. **8-DFB-per-program limit.** Not hit by matmul/eltwise in MVP. If a multi-op fusion blows past it, the conversion pass must fail loudly with a clear message — out of scope to implement spill/reuse, in scope to add the guard.

6. **`Arch::Quasar` enum absence.** Plan uses an explicit `use-dfbs` flag; revisit once Quasar enum lands.

7. **Future MPMC will likely re-introduce D2M-level DFB IR.** When nP-side cardinality matters for scheduling decisions (e.g., choosing nP→1 layout for a fan-in pattern), the D2M layer will need to carry cardinality on memrefs. That work would re-introduce something like `#ttcore.dfb_layout` and possibly an `!d2m.dfb` vocabulary. For the current MVP it would be premature.

## Verification

End-to-end:
- `source env/activate && cmake --build build --target check-ttmlir` — full lit suite green; new `*_dfb.mlir` conversion tests pass; existing CB tests remain bit-identical (no D2M passes touched, so CB output is verbatim).
- `cmake --build build && ttrt query --save-artifacts` (Quasar system) and `SYSTEM_DESC_PATH=...` set, then `ttmlir-opt --ttir-to-ttmetal-pipeline="use-dfbs=true ..." matmul.mlir | ttrt run -` against Quasar silicon (or tt-emule).
- `pre-commit run --all-files`.

Specific FileCheck guards:

```
// DFB path with flag:
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-dfbs=true" %s | FileCheck %s --check-prefix=DFB
// DFB: ttmetal.create_dataflow_buffer
// DFB: ttmetal.bind_dfb_to_kernels
// DFB: ttkernel.dfb_reserve_back
// DFB: ttkernel.dfb_wait_front
// DFB: ttkernel.dfb_finish
// DFB-NOT: ttkernel.cb_reserve_back

// CB path preserved without flag (regression guard):
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline %s | FileCheck %s --check-prefix=CB
// CB: ttkernel.cb_reserve_back
// CB-NOT: ttkernel.dfb_reserve_back

// 1P->4C produces a single compute kernel, num_consumers=4 in dfb_config:
// CHECK: func.func @compute_kernel_0
// CHECK-NOT: func.func @compute_kernel_1
// CHECK: ttmetal.create_dataflow_buffer
// CHECK-SAME: num_consumers = 4
// CHECK: ttmetal.bind_dfb_to_kernels
// CHECK-SAME: consumer_kernel = @compute_kernel_0
```

Runtime arch-gating: a runtime unit test loads a DFB-bearing flatbuffer on a mock WH/BH device and asserts the executor returns an error message containing `"DFB"` and `"Quasar"`.

Silicon golden references: mirror the simplest "1P→1C explicit-sync reserve/push/wait/pop" and "1P→nC BLOCKED broadcast" tests from `tt-emule/run_regression.sh` as the trusted reference numerics.

## Appendix: revised approach vs. earlier plan

The earlier version of this plan committed to a **parallel D2M vocabulary** — a full set of `!d2m.dfb`, `d2m.get_dfb`, `d2m.dfb_*` ops, plus a `#ttcore.dfb_layout` memref attribute, threaded through `D2MAllocate` / `D2MHoistCBAllocs` / `D2MMaterializeComputeThreadForall` / `D2MNormalizeThreadArgs`. PR1 (`40ca13be0`), PR2 (`8dabb8844`), and PR3 (`6839c60be`) of that earlier sequence have already landed on `arminale/quasar-tiling`.

The earlier design's load-bearing assumption was that the consumer-side compute kernel needed a per-thread DFB consumer-slot SSA value plumbed from `d2m.my_thread_id` into `d2m.dfb_wait`, surviving lowering. That assumption is wrong: the Quasar `DataflowBuffer` kernel API takes no consumer-slot operand; the slot is established host-side at bind time and selected on-device via `mhartid`. Once that's true, the D2M middleend has nothing to gain from carrying DFB-vs-CB information, and threading a parallel vocabulary through the middleend is purely overhead.

The revised approach is to roll back PR1's `#ttcore.dfb_layout` plus all of PR2, keep PR3's TTKernel + TTMetal surface (which is genuinely needed), and concentrate the DFB-vs-CB choice in `ConvertD2MToTTKernel` + `ConvertD2MToTTMetal`. The total remaining work is roughly half of what the earlier plan called for.
