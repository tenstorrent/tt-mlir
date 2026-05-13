# D2M DFB Codegen Bring-Up (Phase 1: SPSC + 1P→nC compute)

## Context

Quasar's Tensix Neo replaces the Wormhole/Blackhole **Circular Buffer (CB)** primitive with the **Dataflow Buffer (DFB)** — an MPMC FIFO backed by per-consumer hardware tile counters with explicit `posted`/`acked` pairs. CBs do not exist on Quasar; targeting Quasar requires the compiler to emit DFB host setup (`CreateDataflowBuffer`, `BindDataflowBufferToProducerConsumerKernels`) and DFB kernel ops (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`finish`) in place of CB ops.

D2M today emits only SPSC dataflow (1 DM producer → 1 compute consumer through one CB). Bringing up DFBs has two largely independent parts: (1) API retargeting — emit `ttkernel.dfb_*` and host setup ops in place of `ttkernel.cb_*`; (2) IR generalization — let the IR express MPMC topologies. This plan covers Part 1 plus the smallest slice of Part 2 needed for the existing 4-compute-thread matmul (1P→nC, BLOCKED on the consumer side).

The existing scoping doc at `docs/src/specs/d2m-dataflow-buffer-bringup.md` argued for replacing `!d2m.cb` with `!d2m.dfb` universally. This plan deliberately takes the **parallel-vocabulary, target-gated** approach instead, per the user's explicit choice — DFBs only emit when targeting Quasar, CB path is preserved bit-for-bit on Wormhole/Blackhole. The two vocabularies coexist; lowering picks one based on a pipeline flag (`use-dfbs`), to be replaced by arch-driven selection once a `Quasar` `Arch` enum lands.

**Scope (in this plan):**
- Ops: matmul (existing 4-compute-thread tiling: `enable-compute-thread-tiling=true num-compute-threads=4`), eltwise unary, eltwise binary.
- Topologies: 1P→1C (drop-in for CB), 1P→nC where nC = compute-thread count (BLOCKED on the consumer side, broadcast input).
- Sync mode: explicit (`reserve_back`/`push_back`/`wait_front`/`pop_front` + `finish`). No implicit transaction-ID sync.
- End-to-end: compiler IR + TTMetal/flatbuffer schema + runtime executor + Quasar silicon test.

**Out of scope:** nP→1C, nP→nC, STRIDED consumer patterns, implicit sync, reductions, the per-program 8-DFB-limit fallback. These are Phases 3–4 in the scoping doc.

**Key constraint:** The N compute threads remain a single `scf.forall` (already in IR) lowered by `D2MMaterializeComputeThreadForall` to one SPMD compute kernel using `d2m.my_thread_id`. The compute kernel binary must be **identical across all N compute threads** — no per-thread specialization, no kernel-function unrolling. The runtime DFB binding selects which TC slot each compute thread reads, driven by `mhartid` on device (which `my_thread_id` lowers to).

## Approach

### A. IR design — parallel DFB vocabulary

Add a sibling vocabulary that mirrors CB; do **not** modify the existing CB ops/types/attrs.

**TTCore attrs/enums** (`include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td`, `TTCoreOpsEnums.td`):
- `#ttcore.dfb_layout<stride, num_entries, num_producers, num_consumers, producer_pattern, consumer_pattern>` (MemRefLayoutAttrInterface, mirrors `CBLayoutAttr`).
- Enum `TTCore_DFBAccessPattern { STRIDED, BLOCKED }`.

**D2M layer** (`include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td`, `D2MOpsTypes.td`):
- New type `D2M_DFB` (analogue of `D2M_CB`).
- New ops `D2M_DFB{Wait,Reserve,Push,Pop,Finish}Op` and `D2M_GetDFBOp`. `GetDFBOp` carries the same `cb_operand_idx` and `resolution_stage` as `GetCBOp`, plus an optional `consumer_slot` SSA operand (an `index`) and a `num_consumers` attribute.
- Reuse the `D2MGenericRegionOp` trait wrappers and verifiers from the CB side; copy, don't refactor.

**TTKernel layer** (`include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td`, `TTKernelOps.td`):
- New type `!ttkernel.dfb<num_entries, element_type, num_producers, num_consumers>`.
- New ops `ttkernel.dfb_{reserve_back, push_back, wait_front, pop_front, finish}`. `dfb_wait_front` carries `TTKernel_DeviceZoneOpTrait` (parity with `cb_wait_front`).
- New `ArgType::DFBId` in `TTKernelOpsEnums.td` (mirrors `ArgType::CBPort`).
- The kernel-side ABI does **not** take a consumer-slot argument on `wait_front`/`pop_front`; the slot is established at host-side bind time. The MLIR op carries the `consumer_slot` SSA value for verification/analysis only; it is dropped by the EmitC rewriter.

**TTMetal host-side layer** (`include/ttmlir/Dialect/TTMetal/IR/TTMetalOps.td`, `TTMetalOpsAttrs.td`):
- `#ttmetal.dfb_config<entry_size, num_entries, num_producers, num_consumers, producer_risc_mask, consumer_risc_mask, producer_pattern, consumer_pattern, enable_implicit_sync=false, data_format>`.
- `ttmetal.create_dataflow_buffer(core_range, config) -> ui32` — returns the logical DFB id.
- `ttmetal.bind_dfb_to_kernels(dfb_id, producer_kernel: symbol, consumer_kernel: symbol)`.
- Extend `TTMetal_EnqueueProgramOp` with a parallel `Variadic<...>:$dfbs` + `DenseI64ArrayAttr:$dfb_ids` operand list, alongside the existing `$cbs` + `$cb_ports`.

### B. Pass changes

Pipeline reference: `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp`.

1. **`D2MAllocate`** (`lib/Dialect/D2M/Transforms/Allocate.cpp` — `requiresCBAllocation`, `getCBBufferType`, `operandCBTypeByIndex` ~L634, L885, L910, L1339, L1422). New `dfbAllocateMode` option on `D2MAllocateOptions`. When on, stamp `#ttcore.dfb_layout` in place of `#ttcore.cb_layout`. Policy for the MVP:
   - `num_producers = 1` (the producer is always a single DM thread on this branch).
   - `num_consumers = numComputeThreads` if the operand is read by a compute thread region **and** the operand is logically broadcast across compute threads (BLOCKED). For sliced-per-thread operands, `num_consumers = 1`.
   - Source of signal: walk the parent `d2m.generic`'s thread regions, classify per-operand by `ThreadType`, and consult `enableComputeThreadTiling` / `numComputeThreads` already in `D2MPipelineOptions`.
   - For eltwise, all operands are 1P→1C.
   - `num_entries` defaults to `2` for 1P→1C, `max(2, numComputeThreads)` for BLOCKED 1P→nC.

2. **`D2MHoistCBAllocs`** (`lib/Dialect/D2M/Transforms/HoistCBAllocs.cpp` ~L33–98). Generalize the predicate at L40 from `isa<CBLayoutAttr>` to `isa<CBLayoutAttr, DFBLayoutAttr>`. Rename to `D2MHoistBufferAllocs`; keep the old name as a deprecated alias for one release cycle. Wire the new name in `D2MPipelines.cpp:234`.

3. **`D2MMaterializeComputeThreadForall`** (`lib/Dialect/D2M/Transforms/MaterializeComputeThreadForall.cpp`). After splicing the forall body (already inserts `d2m.my_thread_id` ~L60), for every `d2m.get_dfb` inside the consumer-side (compute thread) block whose `consumer_slot` operand is absent, set it to the `my_thread_id` value. This is the seam that gives every compute thread its TC index without specializing the kernel binary.

4. **`D2MNormalizeThreadArgs`** (`lib/Dialect/D2M/Transforms/NormalizeThreadArgs.cpp` ~L70–105). Add a `Case<GetDFBOp>` branch in the resolution-stage type-switch that mirrors the existing `GetCBOp` branch. Make the additional-arg rebuild logic skip DFB-typed args (parallel to how CB args are skipped).

5. **`ConvertD2MToTTKernel`** (`lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` — CB patterns at ~L2794–2821 for `GetCBOp`, ~L3251 for `D2MCBOpRewriter` family). Add:
   - `D2MGetDFBRewriter`: appends an `ArgType::DFBId` to the function's `ArgSpec`, emits `ttkernel.get_compile_time_arg_val(ctIdx)` to retrieve `logical_dfb_id`, and a small adapter `ttkernel.dfb_from_id` op (or inline `unrealized_conversion_cast`) to produce a typed `!ttkernel.dfb` SSA value.
   - `D2MDFBOpRewriter` family for `DFBWait/Reserve/Push/Pop/Finish` → corresponding `ttkernel.dfb_*` ops. Producer-side patterns mirror existing CB push patterns; consumer-side mirror existing CB pop patterns. The `consumer_slot` operand is consumed in the rewriter for verification but is **not** passed to the TTKernel op for emission.
   - Verifier: every DFB-typed value with producer-side ops must have at least one terminating `dfb_finish`. Failure to emit `finish` would silently hang on Quasar.

6. **`ConvertD2MToTTMetal`** (`lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` ~L199, L237, L295, L322–324).
   - In `MemrefAllocRewriter`, branch on `DFBLayoutAttr`: emit `ttmetal.create_dataflow_buffer` instead of `ttmetal.create_buffer`, building `DataflowBufferConfigAttr` from the layout's fields plus derived `producer_risc_mask` / `consumer_risc_mask` (small utility helper; see Risks below).
   - In the `EnqueueProgramOp` builder, route DFB-typed additional-args into the new `$dfbs` / `$dfb_ids` operands.
   - For each DFB, emit `ttmetal.bind_dfb_to_kernels` referencing the producer kernel func symbol and the consumer kernel func symbol. (For 1P→4C, both 4 compute threads share the same kernel symbol — bind is one-to-one by symbol; the runtime allocates 4 TC slots from `num_consumers=4` in the config.)

7. **`ConvertTTKernelToEmitC`** (`lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` ~L1362). Add opaque rewriters that emit:
   - At kernel-fn prologue (once per DFB id used): `experimental::DataflowBuffer dfb_<id>(static_cast<uint16_t>(<get_compile_time_arg_val>));`
   - `dfb.reserve_back(n);`, `dfb.push_back(n);`, `dfb.wait_front(n);`, `dfb.pop_front(n);`, `dfb.finish();`
   - Use the `DataflowBuffer(uint16_t logical_dfb_id)` constructor form (not `DFBAccessor`) — single ct_args slot, parallel to how CB ports are plumbed today via `KernelArgCBPort`.

### C. Pipeline option

Add to `include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h::D2MPipelineOptions`:

```
Option<bool> useDFBs{*this, "use-dfbs",
    llvm::cl::desc("Emit dataflow buffers (DFBs) instead of circular buffers; required for Quasar."),
    llvm::cl::init(false)};
```

Plumbed into `D2MAllocateOptions.dfbAllocateMode`, `ConvertD2MToTTKernelOptions.useDFBs`, `ConvertD2MToTTMetalOptions.useDFBs`. CLI: `--ttir-to-ttmetal-pipeline="use-dfbs=true ..."`. Default off — `Arch::Quasar` is not yet defined in `TTCoreOpsEnums.td`; arch-driven defaulting is a follow-up once that lands.

### D. Host-side / flatbuffer / runtime

Flatbuffer schema (`include/ttmlir/Target/TTMetal/`):
- `types.fbs`: add `enum DFBAccessPattern : ushort { STRIDED, BLOCKED }`, `table DataflowBufferConfig {...}`, `table DFBRef { global_id, dfb_config, core_range_set }` (mirrors `CircularBufferConfig` / `CBRef` at L40 area).
- `command.fbs`: add `table CreateDataflowBufferCommand { ref: DFBRef; }` and `table BindDFBToKernelsCommand { dfb_global_id, producer_kernel_symbol, consumer_kernel_symbol }`; add both to the `CommandType` union (~L92).
- `program.fbs`: add `KernelArgDFBId { operand_idx: uint32 }`; add to `KernelArgType` union (~L88, mirrors `KernelArgCBPort` at L63).
- `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp`: serialize the new ops.

Runtime (`runtime/lib/ttmetal/`):
- `executor.cpp` (CreateBufferCommand handler at ~L493, EnqueueProgramCommand at ~L355, createCircularBufferConfig at ~L434): add `CreateDataflowBufferCommand` handler calling `experimental::dfb::CreateDataflowBuffer`, `BindDFBToKernelsCommand` handler calling `experimental::dfb::BindDataflowBufferToProducerConsumerKernels`. Maintain a per-program DFB id table keyed by `global_id`. Resolve consumer/producer `KernelHandle` from symbol via the program's kernel registry already populated by `EnqueueProgramCommand`.
- `executor.h`, `executor_utils.h`, `arguments.h`, `kernels.{cpp,h}`: parallel additions to support `KernelArgDFBId` ct_args plumbing.
- **Arch gating in runtime:** if `CreateDataflowBufferCommand` dispatches against a non-Quasar device, emit a runtime error `"DFB ops require Quasar"` rather than silently corrupting. The compiler's `use-dfbs=true` on a non-Quasar target then fails loudly at `ttrt run`.

### E. Tests

Lit (under `test/ttmlir/`):
- `Dialect/D2M/Transforms/allocate_dfb.mlir` — `--d2m-allocate=use-dfbs=true`: verify `#ttcore.dfb_layout` appears, `#ttcore.cb_layout` does not.
- `Dialect/D2M/Transforms/hoist_buffer_allocs.mlir` — verify both CB and DFB allocs hoist; rename test to match the new pass name.
- `Dialect/D2M/Transforms/materialize_compute_thread_forall_dfb.mlir` — verify `d2m.get_dfb` carries `d2m.my_thread_id` SSA value as `consumer_slot` on compute side.
- `Dialect/D2M/Transforms/normalize_thread_args_dfb.mlir` — verify resolution-stage propagation on `d2m.get_dfb`.
- `Conversion/D2MToTTKernel/d2m_to_ttkernel_dfb.mlir` — verify `ttkernel.dfb_*` ops and `ArgType::DFBId` ArgSpec.
- `Conversion/D2MToTTMetal/d2m_to_ttmetal_dfb.mlir` — verify `ttmetal.create_dataflow_buffer` + `ttmetal.bind_dfb_to_kernels` for both 1P→1C and 1P→4C.
- `Conversion/TTKernelToEmitC/ttkernel_to_emitc_dfb.mlir` — verify emitted `DataflowBuffer dfb(id); dfb.reserve_back(...)` form.
- **Regression guard:** existing matmul/eltwise lit tests without `use-dfbs` must remain bit-identical to before.

Silicon end-to-end (under `test/ttmlir/Silicon/D2M/quasar/`):
- `eltwise_unary.mlir` — 1P→1C.
- `eltwise_binary.mlir` — 2× 1P→1C input + 1× 1C→1P output.
- `matmul_4ct.mlir` — `use-dfbs=true,enable-compute-thread-tiling=true,num-compute-threads=4`; 1P→4C input DFB for the broadcast operand.

### F. Suggested PR sequence

1. **IR scaffolding** — `#ttcore.dfb_layout`, `D2M_DFB`/`!ttkernel.dfb`/`DataflowBufferConfig` types/attrs. Parser round-trip tests only. (Files: 4 `.td` files.)
2. **D2M sync ops** — `D2M_DFB{Wait,Reserve,Push,Pop,Finish}Op`, `D2M_GetDFBOp`. Verifier + parser tests.
3. **TTKernel + TTMetal surface** — `ttkernel.dfb_*` ops, `ttmetal.create_dataflow_buffer`, `bind_dfb_to_kernels`, `EnqueueProgramOp` `dfbs`/`dfb_ids` operands. No conversions yet.
4. **D2MAllocate + HoistBufferAllocs behind `use-dfbs`** — flag plumbed; allocator stamps DFB layout; hoister generalized. (Files: `Allocate.cpp`, `HoistCBAllocs.cpp`→`HoistBufferAllocs.cpp`, `Passes.td`, `D2MPipelines.{h,cpp}`.)
5. **`ConvertD2MToTTKernel` DFB patterns** — `GetDFBRewriter`, `DFBOpRewriter` family. (File: `D2MToTTKernel.cpp`.)
6. **`ConvertD2MToTTMetal` + `MaterializeComputeThreadForall` consumer-slot threading** — emits host create/bind ops; threads `my_thread_id` into `get_dfb`. (Files: `D2MToTTMetal.cpp`, `MaterializeComputeThreadForall.cpp`.)
7. **`ConvertTTKernelToEmitC` DFB rewriters** — final C++ emission. (File: `TTKernelToEmitC.cpp`.)
8. **Flatbuffer schema + runtime executor** — end-to-end silicon path. (Files: `Target/TTMetal/types.fbs`, `command.fbs`, `program.fbs`, `TTMetalToFlatbuffer.cpp`, `runtime/lib/ttmetal/{executor.cpp,executor.h,executor_utils.h,arguments.h,kernels.{cpp,h}}`.)

Each of PRs 1–4 is independently mergeable as surface additions or behind a default-off flag; CB tests remain green. PRs 5–7 each fill one conversion layer and ship with lit tests; the chain isn't usable end-to-end until PR8 closes the runtime path.

Follow-up (out of this plan): add `Arch::Quasar` and switch `use-dfbs` to arch-driven default; nP→nC topologies; implicit-sync mode; per-program 8-DFB-limit fallback.

## Critical files

**Modified:**
- `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td`, `TTCoreOpsEnums.td`
- `include/ttmlir/Dialect/D2M/IR/D2MOpsTypes.td`, `D2MGenericRegionOps.td` (~L2335–2463 CB region ops)
- `include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h`
- `include/ttmlir/Dialect/D2M/Transforms/Passes.td`
- `include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td`, `TTKernelOps.td` (~L2874–2931 CB ops), `TTKernelOpsEnums.td`
- `include/ttmlir/Dialect/TTMetal/IR/TTMetalOps.td`, `TTMetalOpsAttrs.td`
- `include/ttmlir/Target/TTMetal/types.fbs`, `command.fbs`, `program.fbs`
- `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` (~L75–271 frontend/backend pipelines)
- `lib/Dialect/D2M/Transforms/Allocate.cpp` (CB stamping ~L885/L910, `requiresCBAllocation` ~L634)
- `lib/Dialect/D2M/Transforms/HoistCBAllocs.cpp` → `HoistBufferAllocs.cpp`
- `lib/Dialect/D2M/Transforms/MaterializeComputeThreadForall.cpp` (my_thread_id insertion ~L60)
- `lib/Dialect/D2M/Transforms/NormalizeThreadArgs.cpp` (~L70–105)
- `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` (`GetCBRewriter` ~L2794–2821, `CBOpRewriter` family ~L3251)
- `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` (~L199, L237, L295, L322–324)
- `lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` (~L1362)
- `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp`
- `runtime/lib/ttmetal/executor.{h,cpp}` (CreateBuffer ~L493, EnqueueProgram ~L355/L434), `executor_utils.h`, `arguments.h`, `kernels.{cpp,h}`

**Reused existing utilities:**
- CB allocation/hoist/normalize/conversion patterns serve as line-by-line templates for DFB analogues.
- `D2MGetCBRewriter` at `D2MToTTKernel.cpp:2794` is the model for `D2MGetDFBRewriter`.
- `D2MCBOpRewriter<...>` template at `D2MToTTKernel.cpp:3251` is the model for `D2MDFBOpRewriter<...>`.
- Existing TTKernel `CBPort` ArgType plumbing is the model for `DFBId` ArgType plumbing.
- Existing `MemrefAllocRewriter` in `D2MToTTMetal.cpp:295` is the model for `create_dataflow_buffer` emission.
- `experimental::DataflowBuffer` device API at `/localdev/arminale/tt-metal-main/tt_metal/hw/inc/experimental/dataflow_buffer.h`.
- Host API at `/localdev/arminale/tt-metal-main/tt_metal/api/tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp`.
- Existing scoping doc `docs/src/specs/d2m-dataflow-buffer-bringup.md` — note this plan diverges on the parallel-vs-replace question (parallel) and limits scope to Phases 1–2.

## Risks / unknowns

1. **`risc_mask` derivation.** `DataflowBufferConfig` needs `producer_risc_mask` and `consumer_risc_mask` (DM RISCs in bits 0–7, Tensix RISCs in bits 8–15). Add a single utility `deriveDfbRiscMask(KernelFuncOp, role)` that walks the kernel func's thread type / id and returns the bitmask; unit-test it standalone. Getting this wrong silently mis-binds the DFB and hangs.
2. **`dfb_finish` placement.** Missing `finish` is a silent Quasar hang. Add a verifier in `ConvertD2MToTTKernel` that asserts every DFB-typed value with producer-side ops has at least one `dfb_finish` on every kernel-exit path.
3. **TTMetal symbol resolution at runtime.** `bind_dfb_to_kernels` references kernels by symbol; the runtime resolves to `KernelHandle` via the program's kernel registry. Verify the registry is populated in dispatch order such that `BindDFBToKernelsCommand` can resolve symbols when it runs (likely needs to dispatch after kernel creation but before program launch — same ordering CB ports already rely on).
4. **8-DFB-per-program limit.** Not hit by matmul/eltwise in MVP. If a multi-op fusion blows past it, allocator must spill — out of scope for this plan but worth a guard error in `D2MAllocate` that fails loudly with a clear message.
5. **`Arch::Quasar` enum absence.** Plan uses a flag (`use-dfbs`) instead of arch detection. Acceptable for MVP; revisit once Quasar enum lands.
6. **Existing scoping-doc divergence.** This plan chooses parallel vocabulary; the scoping doc leaned toward replace. The user's explicit choice resolves this. Document the divergence in PR1's commit message so reviewers don't bounce on the inconsistency.

## Verification

End-to-end:
- `source env/activate && cmake --build build --target check-ttmlir` — full lit suite green; new `dfb_*.mlir` tests pass; existing CB tests remain bit-identical.
- `cmake --build build && ttrt query --save-artifacts` (Quasar system) and `SYSTEM_DESC_PATH=...` set, then `ttmlir-opt --ttir-to-ttmetal-pipeline="use-dfbs=true ..." matmul.mlir | ttrt run -` against Quasar silicon (or tt-emule).
- `pre-commit run --all-files`.

Specific FileCheck guards in lit tests:

```
// DFB path with flag:
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-dfbs=true" %s | FileCheck %s --check-prefix=DFB
// DFB: ttmetal.create_dataflow_buffer
// DFB: ttmetal.bind_dfb_to_kernels
// DFB: ttkernel.dfb_reserve_back
// DFB: ttkernel.dfb_wait_front
// DFB: ttkernel.dfb_finish
// DFB-NOT: ttkernel.cb_reserve_back

// CB path preserved without flag (regression guard on existing matmul test):
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline %s | FileCheck %s --check-prefix=CB
// CB: ttkernel.cb_reserve_back
// CB-NOT: ttkernel.dfb_reserve_back

// Identical compute-kernel binary for 1P→4C:
// CHECK: func.func @compute_kernel_0
// CHECK-NOT: func.func @compute_kernel_1
// CHECK: %[[TID:.+]] = d2m.my_thread_id
// CHECK: d2m.get_dfb({{[0-9]+}}, %[[TID]])

// Runtime arch-gating (runtime unit test):
// EXPECT: error message containing "DFB" and "Quasar" when running a DFB-bearing flatbuffer on a WH/BH device.
```

Silicon golden references: mirror the simplest "1P→1C explicit-sync reserve/push/wait/pop" and "1P→nC BLOCKED broadcast" tests from `/localdev/arminale/tt-emule/run_regression.sh` as the trusted reference numerics for the three end-to-end silicon tests.
