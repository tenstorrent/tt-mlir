# Done report — initial 2-NoC bring-up run (2026-05-19)

> **Snapshot taken before reverting `7e29a386a`.** This report
> describes the first end-to-end Quasar matmul codegen bring-up, which
> was built on the wrong premise that Quasar has 2 NoCs. The Phase B
> ScheduleDMA fix in that commit is being reverted; downstream fixes
> stay. The corrected run with `num-nocs=1` will produce a separate
> report.

## Phase B — pipeline runs end-to-end

Both `matmul_tile` and `matmul_block` variants compiled through
`ttir-to-ttmetal-pipeline` with the Quasar flag combo
`use-dfbs=true num-datamovement-processors=6 enable-compute-thread-tiling=true num-compute-threads=1`
(NoC count left at its 2-NoC default — **the false premise**).

Four compiler fixes landed during bring-up (each as one intermediate
commit on `arminale/quasar-tiling`, none pushed):

1. **`7e29a386a` ScheduleDMA: support 2 NoCs with >2 datamovement processors**
   The 2-NoC scheduler hard-capped at 2 DM threads. Removed the cap and
   added a 2-NoC + N-DM round-robin assignment.
   **— TO BE REVERTED. The premise is wrong: Quasar has a single NoC.
   The existing `num-nocs=1` path already supports N DMs.**
2. **`159085352` Plumb DFB through conversion and EmitC for multi-DM Quasar**
   - Drop the up-front rejection of explicit DM processor indices in
     D2MToTTKernel / D2MToTTMetal.
   - `D2MGetArgRewriter`: accept DFB-typed converted memrefs (not just
     CB), emitting DFBId ct_args; otherwise BufferAddress fell through.
   - `classifyCBRole`: scan ALL ct_arg indices matching a CB operand,
     not just the first — the compute kernel binds the same DFB twice
     via two `d2m.get_cb` calls, and only one carried the wait/pop ops.
   - `TTKernelToEmitC.getCBName` / `ensureCBDeclaration`: emit
     `experimental::DataflowBuffer dfb_ctarg_N` for DFB-typed values.
3. **`6c434091d` Make EmitC output compile against Quasar headers**
   - `TTKernelToEmitCOpaqueRewriter` wraps DFB operands of free-function
     compute calls (mm_init, matmul_tiles, pack_tile, etc.) in
     `dfb_ctarg_N.get_id()` to satisfy the `uint32_t` ABI.
   - `TTKernelDFBMethodRewriter` routes the method-call receiver through
     `ensureCBDeclaration` so DFB sync ops call into the declared
     DataflowBuffer object instead of the raw ct_arg literal.
   - `TTKernelToCpp` auto-includes `experimental/dataflow_buffer.h`.
4. **`6aab42f41` enqueue_program: omit cb operands under useDFBs**
   After emitDFBHostOps populates the DFB operand groups, clear cbs /
   cb_ports so the resulting program is DFB-only. tt-metal asserts on
   mixing the two within one program ("Cannot add circular buffer to a
   program that already has dataflow buffers"); the conversion was
   populating both with the same buffer set.

All 431 lit tests under `test/ttmlir/Conversion/` and
`test/ttmlir/Dialect/D2M/` pass after every fix.

## Phase C — 7-item structural checklist

Both variants pass all 7 checks identically; only item 5 (compute body)
differs:

| Check | tile | block |
|---|---|---|
| 1. 9 kernel funcs (5 noc + 4 compute) | ✓ | ✓ |
| 2. DFB host setup (9 create + 9 bind) | ✓ | ✓ |
| 3. Zero CB sync ops; 36 DFB method calls | ✓ | ✓ |
| 4. dfb_finish in producer kernels (12 / 11) | ✓ | ✓ |
| 5. Compute body matches expected op | matmul_tiles ✓ | matmul_block ✓ |
| 6. scf.forall N=1 fully materialized | ✓ | ✓ |
| 7. EmitC C++ uses DataflowBuffer (no CB leakage) | ✓ | ✓ |

The matmul d2m.generic emitted 2 DM kernels — `datamovement_kernel4
noc=0 processor=0` (LHS reader) and `datamovement_kernel5 noc=1
processor=1` (RHS reader) — under the (wrong) 2-NoC scheduling. After
the revert + `num-nocs=1` rerun, both DMs are expected on `noc = 0`
with distinct processor indices, mirroring the existing
`schedule_dma_multi_processor.mlir` 1-NoC shape.

## Phase D — kernel C++ compiles standalone

Both variants' compute_kernel6 translates via
`ttmlir-translate --ttkernel-to-cpp` and compiles with:

```
clang-20 -std=c++20 -fsyntax-only \
  -I/localdev/arminale/tt-emule/include \
  -I/localdev/arminale/tt-emule/include/jit_hw \
  -I/localdev/arminale/tt-metal/tt_metal/hostdevcommon/api \
  -DARCH_QUASAR -DKERNEL_COMPILE_TIME_ARGS=0,0,0,0
```

Exit 0, no warnings or errors. Generated code matches the handwritten
Quasar matmul reference (`tt-metal/tests/.../matmul_block.cpp`) shape:
`experimental::DataflowBuffer dfb_ctarg_N(...)` declarations,
`mm_init(dfb.get_id(), …)`, `dfb.wait_front(...)`, `matmul_block` or
`matmul_tiles` inside a K-block accumulation loop, `pack_tile`, then
`dfb.push_back/wait_front/pop_front` and `dfb.finish()` at exit.

## Phase E — execution on tt-emule

Flatbuffer loads cleanly on tt-emule (after the cb/dfb-mix fix in
`6aab42f41`). The matmul compute kernel JIT-compiles and starts
executing.

**Blocked on a real codegen design gap (NoC-independent):** the compute
kernel calls `dfb_reserve_back` on its OUTPUT DFB (the
`d2m.operand_alias` for the matmul output). In the matmul `d2m.generic`
itself there is no downstream consumer for that DFB — the next consumer
is in the following untilize generic, a separate program. tt-emule's
DFB sync expects a producer/consumer pair to drain the counters; with
no consumer the reserve hangs after `TT_EMULE_DFB_TIMEOUT` (default
120s).

The `classifyCBRole` analysis tags the compute kernel as `Both`
(producer + consumer) on this DFB because the matmul kernel both
reserves and pops the same operand_alias buffer for tile addressing
bookkeeping, but the DFB's true downstream consumer lives in a
different program. The Quasar 1P→1C codegen needs to either (a) not
emit dfb_reserve/push_back/wait_front/pop_front on operand_alias-only
outputs, or (b) wire the output DFB to span across the matmul and
untilize programs.

This was out of scope for the validation pass (the original plan
explicitly named the dynamic-CB / DFB host-side limitations as
out-of-scope and authorized the SRAM pre-positioning workaround). The
user-authorized workaround was to also strip the tilize/untilize aux
generics and pre-position L1; that surgery is beyond the four compiler
fixes already landed and is the natural follow-up to drive the matmul
through to PCC against tt-emule.

## Summary

Compiler-level result: **PASS** — Quasar codegen produces structurally
valid, type-correct, compileable matmul kernels for both
`use-tile-matmul=true` and `=false`. Four upstream tt-mlir bugs were
found and fixed locally. Existing lit suites stay green.

Execution result: **partial** — the kernel runs on tt-emule until the
output-DFB design gap is hit. Resolving the gap is follow-up work.

## Critical-files reference (post-bring-up, pre-revert)

- `lib/Dialect/D2M/Transforms/ScheduleDMA.cpp` (extended 2-NoC + N-DM)
  **— TO BE REVERTED**
- `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` (DFB ct_arg routing,
  multi-ct_arg classifyCBRole)
- `lib/Conversion/D2MToTTKernel/D2MToTTKernelPass.cpp` (dropped processor-index reject)
- `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` (cb/dfb mix fix +
  multi-index classifyCBRole)
- `lib/Conversion/D2MToTTMetal/D2MToTTMetalPass.cpp` (dropped processor-index reject)
- `lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` (DFB naming,
  `.get_id()` wrap on free-function operands, DFB method-call receiver)
- `lib/Target/TTKernel/TTKernelToCpp.cpp` (auto-include dataflow_buffer.h)
- `test/ttmlir/Conversion/TTKernelToEmitC/dfb.mlir` (updated for
  declared-object form)
- `test/ttmlir/Dialect/D2M/Transforms/schedule_dma_multi_processor_2noc.mlir`
  (new) **— TO BE DELETED**

## Commit stack (`arminale/quasar-tiling`)

```
6aab42f41 [D2M][DFB] enqueue_program: omit cb operands under useDFBs
6c434091d [TTKernel][DFB] Make EmitC output compile against Quasar headers
159085352 [D2M][DFB] Plumb DFB through conversion and EmitC for multi-DM Quasar
7e29a386a [D2M] ScheduleDMA: support 2 NoCs with >2 datamovement processors  ← reverting
```

Nothing pushed in any repo.
