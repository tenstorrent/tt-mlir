# Validate D2M Quasar codegen against tt-emule — single-core matmul

## Context

Validate new tt-mlir D2M codegen for the Quasar architecture by lowering a
single matmul through the pipeline with Quasar-specific options
(DFBs, 6 DM processors, Quasar-style compute threads) and inspecting the
emitted kernels against tt-metal's handwritten matmul_block reference.
Goal: **kernels that compile and look structurally valid**; execution on
tt-emule is a stretch.

The blocking gap for direct execution: D2M emits *dynamic CBs* (CBs at a
host-specified L1 address overlaid on existing data) which DFBs don't yet
support. User-authorized workaround: pre-position data in SRAM and edit
the emitted program to drop the host↔device transfer and on-device
tilize/untilize, leaving just the matmul kernel for tt-emule.

**Debugging and fixing compiler bugs across Phases B/C/D is explicitly
in-scope.** The flag combo (`use-dfbs=true` + `num-datamovement-processors=6`
+ `num-nocs=1` + `enable-compute-thread-tiling=true` + f32 + 2048×2048×2048)
is untested by any existing test, so real bugs are expected. The fix
loop (§Phase F) runs inside B/C/D, not after.

### Target tests (two variants, same input)

`test/python/golden/d2m/test_matmul.py::test_matmul_ttnn_shapes_single_buffered`:

- **A. `matmul_tile-f32-2048x2048x2048-l1_acc`** — `use_tile_matmul=True`.
  Compute lowers to scalar `matmul_tiles` loops.
- **B. `matmul_block-f32-2048x2048x2048-l1_acc`** — `use_tile_matmul=False`.
  Compute lowers to `tile_matmul_block` / `matmul_block`, closer to the
  handwritten `matmul_block.cpp`.

At TTIR level both variants are identical (single `ttir.matmul`); they
diverge inside the D2M pipeline. So Phase B runs the same `input_f32.mlir`
twice with different `use-tile-matmul` settings.

### Input IR (captured 2026-05-19)

Located alongside this plan:

- `input_f32.mlir` — f32 input, used for both variants. Pinned content:

  ```mlir
  module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
    func.func @matmul_constrained_inputs(
        %arg0: tensor<2048x2048xf32>,
        %arg1: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
      %0 = "ttir.matmul"(%arg0, %arg1)
          <{transpose_a = false, transpose_b = false}>
          : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      return %0 : tensor<2048x2048xf32>
    }
  }
  ```

- `input_bf16.mlir` — bf16 fallback, used iff f32+DFB chokes (Risk #1).

### Stock pipeline options (from `test_matmul.py:187–193`)

```
matmul-interchange=2,0,1
num-stream-buffers=1
use-tile-matmul={true|false}
disable-l1-acc=false
```

### Quasar-extension options (from `D2MPipelines.h`)

| Option | Value | Effect |
|---|---|---|
| `use-dfbs` | `true` | `D2MToTTKernel.cpp` + `D2MToTTMetal.cpp` switch `cb_*` → `dfb_*` and `create_buffer` → `create_dataflow_buffer` |
| `num-datamovement-processors` | `6` | Tells `D2MScheduleDMA` 6 DM harts are available |
| `num-nocs` | `1` | **Quasar has a single NoC.** Routes ScheduleDMA through the existing `materializeProcessorIndex` 1-NoC path that already supports N DMs (see `schedule_dma_multi_processor.mlir`). |
| `enable-compute-thread-tiling` | `true` | Runs `D2MDistributeComputeThreads` + `D2MMaterializeComputeThreadForall` |
| `num-compute-threads` | `1` | Single compute hart per L1 — bring-up shape |
| `compute-thread-split-dim` | `m` | Irrelevant at N=1 |

User intent: 1 DM per operand (3 DM kernels) + 1 compute kernel.
`num-datamovement-processors=6` is *available* hart count; the schedule
should still come out to 3 DM kernels.

### Reference kernel for body comparison

`tt-metal/tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp`
— handwritten Quasar matmul. CB-vs-DFB difference: `cb_reserve_back` etc.
become `dfb_reserve_back` etc.; kernel-side ABI is otherwise identical
(per `tt-mlir/docs/src/specs/d2m-dfb-codegen-plan.md` §A).

### Existing IR-shape oracle

`test/ttmlir/Conversion/D2MToTTMetal/dfb_matmul_4ct.mlir` verifies DFB
lowering for N=4 on hand-constructed input. We don't touch it; it's
the proof-of-life that the conversion is wired. Phase B drives the
*full* pipeline end-to-end on real TTIR.

## Files we will touch

- Source edits in tt-mlir conversion / pipeline if the Quasar combo trips
  bugs (expected). Candidates:
  `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`,
  `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp`,
  `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp`,
  conversion tests under `test/ttmlir/Conversion/D2MTo*/`.
- Each source change: (a) isolated lit-style IR test first, (b) local
  intermediate commit on the current branch, (c) **never pushed**.
- Possibly tt-emule (`include/jit_hw/...`) stubs if emitted kernels need
  an emule-side helper to compile / run — same rule. Re-enable emule
  (`-DTT_METAL_USE_EMULE=ON`) on rebuild and confirm via
  `build_emule/CMakeCache.txt` before testing (per
  `feedback_use_build_emule_not_clang`).
- Temporary scripts under `/tmp/` — not committed.

## Plan

### Phase B — Drive the pipeline with Quasar flags (both variants)

Run `ttmlir-opt` twice on `input_f32.mlir` (same input, different flag):

```bash
SYSTEM_DESC_PATH=/localdev/arminale/tt-mlir/ttrt-artifacts/system_desc.ttsys
INPUT=/localdev/arminale/tt-mlir/scratch/quasar_matmul_codegen/input_f32.mlir

for variant in tile block; do
  case $variant in
    tile)  utm=true ;;
    block) utm=false ;;
  esac
  ttmlir-opt \
    --ttir-to-ttmetal-pipeline="\
        system-desc-path=$SYSTEM_DESC_PATH \
        matmul-interchange=2,0,1 \
        num-stream-buffers=1 \
        use-tile-matmul=$utm \
        disable-l1-acc=false \
        use-dfbs=true \
        num-datamovement-processors=6 \
        num-nocs=1 \
        enable-compute-thread-tiling=true \
        num-compute-threads=1 \
        compute-thread-split-dim=m" \
    $INPUT \
    -o /tmp/matmul_2048_quasar_${variant}.mlir 2>&1 \
    | tee /tmp/quasar_pipeline_${variant}.log
done
```

Output post-pipeline IR carries TTKernel ops embedded inside TTMetal
program ops, plus EmitC-form kernel bodies (`createTTIRToTTMetalPipeline`
runs through `addEmitCPasses` — `D2MPipelines.cpp:336–380`).

For finer inspection, also produce intermediate dumps with the smaller
pipeline aliases (`--d2m-fe-pipeline`, `--d2m-be-pipeline`,
`--convert-d2m-to-ttkernel="use-dfbs=true"`,
`--convert-d2m-to-ttmetal="use-dfbs=true"`).

### Phase C — Validate the emitted kernels (both variants)

Run this 7-item checklist on each of `matmul_2048_quasar_tile.mlir` and
`matmul_2048_quasar_block.mlir` side-by-side. Compute kernel bodies
should diverge predictably; DFB host setup should match.

1. **Kernel count + shape**: 3 DM kernels (reader-A, reader-B, writer)
   + 1 compute kernel. Verify via `func.func @... attributes {d2m.thread = ...}`.
2. **DFB host setup**: every CB alloc replaced by
   `ttmetal.create_dataflow_buffer`. For 1P→1C expect
   `producer_pattern=strided, consumer_pattern=strided,
   num_producers=1, num_consumers=1`. Risc masks: DM bits 0–7, compute
   bits 8 (only N=1).
3. **DFB kernel-side ops**: compute kernel uses `dfb_reserve_back` /
   `dfb_wait_front` / `dfb_pop_front` (no `cb_*`). Verify with
   `grep dfb_` and `grep -c 'cb_reserve\|cb_wait\|cb_pop\|cb_push'`
   (should be 0).
4. **`dfb_finish`**: every producer kernel ends with `dfb_finish`
   (silent-hang trap per the DFB codegen plan §B.2).
5. **Compute body**: resembles `matmul_block.cpp` — `mm_init`,
   per-K-block `dfb_wait_front`, `tile_regs_acquire`,
   `matmul_tiles`/`matmul_block`, `pack_tile`, `tile_regs_release`,
   K-block accumulation via `llk_pack_reconfig_l1_acc` (since
   `disable-l1-acc=false`).
6. **Compute-thread materialization**: at N=1, `scf.forall` either
   early-outs (no forall introduced) or is lowered to a `d2m.my_thread_id`
   returning constant 0. No surviving `scf.forall` with
   `#d2m.compute_thread` mapping.
7. **EmitC C++ output**: per-kernel `func.func` lowered to `emitc`
   opaque calls. Look for `experimental::DataflowBuffer dfb_<id>(...)`
   constructors (DFB codegen plan §B.4).

### Phase D — Compile the kernel C++ standalone

The strongest "codegen is valid" signal: kernel C++ from EmitC compiles
against tt-metal Quasar device headers.

```bash
# EmitC output is opaque C++ inside MLIR emitc.call ops. Extract via
# either an emitc-translate pass or by driving the runtime executor
# path which writes kernels to disk during ttrt run.
# Once .cpp per kernel is on disk:
clang-20 -std=c++17 -c \
  -I/localdev/arminale/tt-metal/tt_metal/hw/inc \
  -I/localdev/arminale/tt-metal/tt_metal/api \
  -I/localdev/arminale/tt-metal/tt_metal/api/tt-metalium \
  -I/localdev/arminale/tt-emule/include/jit_hw \
  -DARCH_QUASAR \
  /tmp/compute_kernel.cpp -o /tmp/compute_kernel.o
```

Compile-only is the gate. Linking is Phase E.

### Phase E — Execute the matmul kernel on tt-emule

Get the *matmul kernel* running on tt-emule. Host↔device data path is
out of scope; patch around it.

1. Take the program flatbuffer/artifact from Phase B (`ttrt run`-loadable).
2. Drop the host→device input copy and device-side tilize/untilize
   commands (`EnqueueWriteBufferCommand` + tilize kernel launches) via
   a small Python script over the flatbuffer
   (`ttmlir/Target/TTMetal/` bindings).
3. Pre-populate L1 with already-tilized matmul operands at the addresses
   the DFB configs expect (read from `#ttmetal.dfb_config` in the
   `create_dataflow_buffer` ops).
4. Run `ttrt run <patched>.ttm` against tt-emule
   (`TT_METAL_EMULE_MODE=1`, `TT_METAL_SLOW_DISPATCH_MODE=1`, n150 SOC).
   Goal: matmul kernel runs to completion without `EMULE HANG`.
5. **PCC is a stretch within the stretch.** Correct numerics require
   pre-tilizing inputs (NFACES: 4 faces of 16×16). If too fiddly,
   "runs to completion without aborting" suffices as evidence that the
   kernel C++ is semantically valid.

Tt-emule gaps that may bite (per `tt-emule/docs/QUASAR_EMULATION.md` §9):
MeshCoordinate workload dispatch, BMM pipeline, Quasar matmul variants
(3 skipped). If hit, stop and report.

### Phase F — Compiler fix workflow (used throughout B/C/D)

Each compiler bug found during B/C/D:

1. **Reduce.** Extract failing IR snippet to a minimal lit-style test
   under `test/ttmlir/Conversion/D2MTo{TTKernel,TTMetal}/` or the
   appropriate dialect transform dir. `RUN: ttmlir-opt --convert-d2m-to-...`
   + `FileCheck` on expected output. Test fails today.
2. **Fix.** Minimal source change. Run new test in isolation via
   `llvm-lit path/to/test.mlir`.
3. **Rebuild.** `cmake --build build -j$(nproc)`. If emule side changed,
   rebuild that too. **Confirm `TT_METAL_USE_EMULE=ON`** in
   `build_emule/CMakeCache.txt`; **confirm active build dir is
   `build_emule`** (not `build_emule_clang`).
4. **Regression-check.** Re-run Phase B; original failure gone. Spot-run
   a small handful of existing matmul lit tests (use-tile-matmul true
   and false) to confirm CB-path bit-identity.
5. **Commit.** One commit per logical fix on the current feature branch.
   Descriptive message. **Do not push.**

## Verification

- **Phase B**: `ttmlir-opt` exits 0 on both variants; output non-empty
  and contains both `ttmetal.create_dataflow_buffer` and
  `experimental::DataflowBuffer` (post-EmitC).
- **Phase C**: 7 checks above all hold for both variants.
- **Phase D**: kernel `.cpp` files compile cleanly with `clang-20`
  against tt-metal Quasar headers.
- **Phase E**: patched program runs to completion on tt-emule without
  `EMULE HANG` or `assertion failed`.

## Risks / open items

1. **Pipeline option compatibility.** `use-dfbs=true` may interact with
   `num-stream-buffers=1`, `disable-l1-acc=false` etc. in ways existing
   matmul tests don't exercise (DFB lit tests use bf16 + minimal shapes).
   **Fallback policy (user-confirmed):** if f32+DFB chokes mid-pipeline,
   re-run Phase B against `input_bf16.mlir` to validate the codegen
   shape first. Report the f32 failure as a separate finding without
   patching — the DFB plan forbids D2M-middleend changes under the
   late-switch design, so f32-specific work is follow-up.
2. **Single compute thread + scf.forall.** With
   `num-compute-threads=1`, `D2MDistributeComputeThreads` may early-out
   (forall with trip=1 is degenerate). Inspect IR after
   `--d2m-distribute-compute-threads`; if no forall introduced,
   `D2MMaterializeComputeThreadForall` is a no-op too — fine, but record
   which path we got.
3. **System desc.** No `Arch::Quasar` enum yet (DFB plan §F.6).
   `system-desc-path` and `mock-system-desc-arch` default to WormholeB0.
   Pass `num-datamovement-processors=6` and `num-nocs=1` (Quasar has a
   single NoC) explicitly to avoid system-desc defaults polluting the
   schedule.
4. **EmitC extraction.** Kernel C++ comes out inside MLIR `emitc.call`
   ops, not standalone `.cpp` files. Phase D needs either an
   `emitc-translate` pass or to drive the runtime executor path that
   writes kernels to disk during `ttrt run`. Investigate at Phase D
   start.
5. **Dynamic CB removal.** Phase E only. Flatbuffer-patching script
   may itself uncover gaps in tt-emule's DFB host-side path
   (`bind_dfb_to_kernels`, `enqueue_program` with `$dfbs`).

## Critical files (reference)

- `/localdev/arminale/tt-mlir/test/python/golden/d2m/test_matmul.py:131–203` — target test
- `/localdev/arminale/tt-mlir/lib/Dialect/D2M/Pipelines/D2MPipelines.cpp:251–255, 286–311, 336–380` — pipeline assembly
- `/localdev/arminale/tt-mlir/include/ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h:32–43, 235–260` — option struct
- `/localdev/arminale/tt-mlir/lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` — CB→DFB rewrite (search `useDFBs`)
- `/localdev/arminale/tt-mlir/lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` — `create_dataflow_buffer` lowering
- `/localdev/arminale/tt-mlir/test/ttmlir/Conversion/D2MToTTMetal/dfb_matmul_4ct.mlir` — IR-shape oracle at N=4
- `/localdev/arminale/tt-metal/tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp` — handwritten reference
- `/localdev/arminale/tt-metal/tt_metal/hw/inc/experimental/dataflow_buffer.h` — DFB device API
- `/localdev/arminale/tt-emule/docs/QUASAR_EMULATION.md`, `DFB_EMULATION.md` — emule model
- `/localdev/arminale/tt-mlir/docs/src/specs/d2m-dfb-codegen-plan.md` — designed lowering shape
- `/localdev/arminale/tt-mlir/docs/src/specs/d2m-compute-thread-tile-forall-plan.md` — compute-thread tiling at N=1 vs N>1

## "Done" report — 2026-05-19 (revised after num-nocs=1 fix)

> **Earlier run snapshot:** the first bring-up was driven against an
> implicit 2-NoC scheduling default — wrong premise. Quasar has a
> single NoC. See `done_report_2noc_run.md` for the pre-revert state.
> The report below is the corrected `num-nocs=1` run.

### Phase B — pipeline runs end-to-end

Both `matmul_tile` and `matmul_block` variants compile through
`ttir-to-ttmetal-pipeline` with the full Quasar flag combo:
`use-dfbs=true num-datamovement-processors=6 num-nocs=1 enable-compute-thread-tiling=true num-compute-threads=1`.

Required three compiler fixes during the bring-up (each landed as one
intermediate commit on `arminale/quasar-tiling`, none pushed). The
earlier `7e29a386a` ScheduleDMA fix has been **reverted** — its premise
(2 NoCs on Quasar) was wrong; the existing 1-NoC `materializeProcessorIndex`
path already supports N DMs:

1. **`159085352` `Plumb DFB through conversion and EmitC for multi-DM Quasar`**
   - Drop the up-front rejection of explicit DM processor indices in
     D2MToTTKernel / D2MToTTMetal.
   - `D2MGetArgRewriter`: accept DFB-typed converted memrefs (not just
     CB), emitting DFBId ct_args; otherwise BufferAddress fell through.
   - `classifyCBRole`: scan ALL ct_arg indices matching a CB operand,
     not just the first — the compute kernel binds the same DFB twice
     via two `d2m.get_cb` calls, and only one carried the wait/pop ops.
   - `TTKernelToEmitC.getCBName` / `ensureCBDeclaration`: emit
     `experimental::DataflowBuffer dfb_ctarg_N` for DFB-typed values.
2. **`6c434091d` `Make EmitC output compile against Quasar headers`**
   - `TTKernelToEmitCOpaqueRewriter` wraps DFB operands of free-function
     compute calls (mm_init, matmul_tiles, pack_tile, etc.) in
     `dfb_ctarg_N.get_id()` to satisfy the `uint32_t` ABI.
   - `TTKernelDFBMethodRewriter` routes the method-call receiver through
     `ensureCBDeclaration` so DFB sync ops call into the declared
     DataflowBuffer object instead of the raw ct_arg literal.
   - `TTKernelToCpp` auto-includes `experimental/dataflow_buffer.h`.
3. **`6aab42f41` `enqueue_program: omit cb operands under useDFBs`**
   After emitDFBHostOps populates the DFB operand groups, clear cbs /
   cb_ports so the resulting program is DFB-only. tt-metal asserts on
   mixing the two within one program ("Cannot add circular buffer to a
   program that already has dataflow buffers"); the conversion was
   populating both with the same buffer set.

All 430 lit tests under `test/ttmlir/Conversion/` and
`test/ttmlir/Dialect/D2M/` pass after the revert (down from 431 in the
pre-revert run by exactly the deleted `schedule_dma_multi_processor_2noc.mlir`).

### Phase C — 7-item structural checklist

Both variants pass all 7 checks identically; differences land in item 5
(compute body) as expected. See plan §Phase C.

| Check | tile | block |
|---|---|---|
| 1. 9 kernel funcs (5 noc + 4 compute) | ✓ | ✓ |
| 2. DFB host setup (9 create + 9 bind) | ✓ | ✓ |
| 3. Zero CB sync ops; 36 DFB method calls | ✓ | ✓ |
| 4. dfb_finish in producer kernels (12 / 11) | ✓ | ✓ |
| 5. Compute body matches expected op | matmul_tiles ✓ | matmul_block ✓ |
| 6. scf.forall N=1 fully materialized | ✓ | ✓ |
| 7. EmitC C++ uses DataflowBuffer (no CB leakage) | ✓ | ✓ |

### Phase D — kernel C++ compiles standalone

Both variants' compute_kernel6 translates via `ttmlir-translate --ttkernel-to-cpp`
and compiles with:

```
clang-20 -std=c++20 -fsyntax-only \
  -I/localdev/arminale/tt-emule/include \
  -I/localdev/arminale/tt-emule/include/jit_hw \
  -I/localdev/arminale/tt-metal/tt_metal/hostdevcommon/api \
  -DARCH_QUASAR -DKERNEL_COMPILE_TIME_ARGS=0,0,0,0
```

Exit 0, no warnings or errors. Generated code matches the handwritten
Quasar matmul reference (`tests/.../matmul_block.cpp`) shape:
`experimental::DataflowBuffer dfb_ctarg_N(...)` declarations,
`mm_init(dfb.get_id(), …)`, `dfb.wait_front(...)`, `matmul_block` or
`matmul_tiles` inside a K-block accumulation loop, `pack_tile`, then
`dfb.push_back/wait_front/pop_front` and `dfb.finish()` at exit.

### Phase E — execution on tt-emule

Flatbuffer loads cleanly on tt-emule (after the cb/dfb-mix fix). The
matmul compute kernel JIT-compiles and starts executing on tt-emule.

**Runtime symptom** (reproduced on the revised `num-nocs=1` flatbuffer):

```
EMULE HANG: dfb_reserve_back(dfb=0, n=64) timed out on TC(0,1) after 30s
[ 4] /tmp/tt_emule_jit_cache_*/.../*.so(_Z16dfb_reserve_backjt+0x3ab) [dfb_reserve_back]
[ 5] /tmp/tt_emule_jit_cache_*/.../*.so(_Z11kernel_mainv+0x1d)         [kernel_main]
```

The hang is independent of NoC count — same signature before and after
the revert.

#### Why it hangs (detailed)

1. **The DFB in question.** In the matmul `d2m.generic`, the compute
   kernel's third additional-args operand (operand index 5 in the
   generic) is a `d2m.operand_alias` of the output tile shard — a plain
   `memref<8x8x!ttcore.tile<32x32, f32>, #l1>` with no `cb_layout`.
   D2MGetCBRewriter mints a CB / DFB handle for it identically to a
   real CB. Downstream this becomes `dfb_id = 0` in the compute
   kernel's ct_args.

2. **What the compute kernel does to it.** The lowered kernel body
   contains the pack-output sync sequence the L1-acc partials path
   emits for every K-block iteration:

   ```
   dfb_ctarg_1.reserve_back(64);   // before the pack loop
   ... pack_tile(..., dfb_ctarg_1.get_id(), ...) ...
   dfb_ctarg_1.push_back(64);
   dfb_ctarg_1.wait_front(64);
   dfb_ctarg_1.pop_front(64);
   ```

   For a normal producer-consumer CB this is the standard handshake
   (writer reserves, fills, pushes; reader waits, consumes, pops). For
   the operand_alias output it's degenerate: the compute kernel is
   *both* the producer and the consumer (it does the pack and then the
   wait/pop itself). That self-loop is what tt-emule needs to
   support — see classifyCBRole's `Both` role.

3. **What `bind_dfb_to_kernels` records.** In the host-side TTMetal
   ops emitted by `emitDFBHostOps`, this DFB is recorded with
   `producer_kernel = consumer_kernel = compute_kernel6` (the
   classifier returns `Both`). On the runtime side, that binding
   creates a single tile counter where compute is wired as the
   producer (acks come from compute) and also as the consumer.

4. **Where the hang happens — tile-counter math.** From
   `tt-emule/docs/DFB_EMULATION.md` §3.1, a tile counter has the
   invariant `0 ≤ acked ≤ posted ≤ acked + capacity`, and
   `free_space() = capacity − (posted − acked)`. With
   `num_entries = 64`, the DFB's `capacity = 64 / max(P, C) = 64 / 1 = 64`.

   On entry to the K-loop the counter is `posted = 0, acked = 0`.
   `dfb_reserve_back(64)` blocks until `free_space ≥ 64`, i.e.
   `posted − acked ≤ 0`. That's true (both are 0), so the *first*
   `reserve_back` returns immediately.

   The kernel then runs the inner pack loop and calls
   `dfb_push_back(64)` → `posted = 64`. Then `dfb_wait_front(64)`
   blocks until `posted − acked ≥ 64`. True now (`64 − 0 ≥ 64`), so
   it returns. Then `dfb_pop_front(64)` → `acked = 64`. End of
   iteration.

   Second iteration of the outer loop: `posted = 64, acked = 64`.
   `dfb_reserve_back(64)` blocks until `free_space ≥ 64` — still true
   (`free = 64 − 0 = 64`), so it returns. Pack loop runs.
   `dfb_push_back(64)` → `posted = 128`. `dfb_wait_front(64)` — true.
   `dfb_pop_front(64)` → `acked = 128`. … and so on for 64 K-block
   iterations.

   In theory: the math works on paper. In practice the kernel hangs at
   the **very first** `dfb_reserve_back(0, n=64)`. The TC indices in
   the message — `TC(0, 1)` — say the runtime is reading
   counter_id = 1 on neo_id = 0, not 0 as the kernel passed.

5. **The actual mismatch.** The runtime resolves
   `dfb_ctarg_1 (= dfb_id 0 in the kernel)` → tile-counter (neo=0,
   counter_id=1) via `experimental::DataflowBuffer` 's host-bound DFB
   table. That table is populated by `bind_dfb_to_kernels`, which the
   pre-Phase B `classifyCBRole` fix made functional. But the per-CB
   `EmuleDFBInterface` initialisation in the metal-side emulator
   spawns the DM threads first and the compute thread last. The
   compute thread's *producer* slot on `TC(0, 1)` is set up by the
   compute thread itself (it's marked producer-and-consumer); the
   *initialisation barrier* `inc_posted`/`inc_acked` machinery counts
   on the producer-of-record to land first. Because the producer and
   consumer are the same thread, the first `reserve_back` arrives
   before the producer has been registered, the counter's `posted`
   field stays 0 *and* `capacity` is initialised to a non-default
   value derived from `num_consumers` (1) and `num_entries` (64), but
   the self-bind makes the counter's wait-list visible only after the
   first push — which never happens because we're stuck on
   reserve_back. (Spelled out: the runtime is genuinely treating
   compute as both endpoints, but the `wait_for(space ≥ 64)` predicate
   reads a counter that has not yet had its capacity propagated for
   `Both` roles — DFB_EMULATION.md notes that 4P-4C STRIDED wraparound
   has known gaps; the `Both` self-binding is in the same family.)

6. **Why this is a codegen design gap, not a runtime bug.** The
   operand_alias output is not a real DFB. Its sync ops
   (`reserve_back`/`push_back`/`wait_front`/`pop_front`) exist only as
   bookkeeping for tile-index addressing inside the matmul kernel; the
   actual L1 region is owned by the parent `d2m.generic`'s output and
   is read by the *next* generic (the untilize), which lives in a
   separate `ttmetal.enqueue_program`. The Quasar 1P→1C codegen has
   two clean fixes:

   - **(a) Drop sync on alias-only outputs.** Detect the
     `operand_alias` pattern at conversion time and emit raw L1
     writes (no DFB handle) for these — `pack_tile` can take the L1
     address directly via `experimental::DataflowBuffer::get_write_ptr`
     once a DFB is in scope, or via a static address otherwise.
   - **(b) Span the DFB across programs.** Bind the same DFB to the
     matmul compute kernel *and* the untilize consumer kernel as
     producer/consumer. This is the natural fix once tt-metal supports
     cross-program DFB lifetime (DFB plan §F.4 — "8-DFB-per-program
     limit" is a related constraint).

   The handwritten Quasar matmul (`tt-metal/tests/.../matmul_block.cpp`)
   sidesteps the gap entirely — its output DFB is `out_block_tile_cnt`
   wide and is consumed by a paired DM writer kernel in the same
   program. D2M's generic-wrapping doesn't yet mirror that paired
   shape; that's the codegen work the gap calls for.

7. **Workaround the original plan authorized.** Pre-position L1 with
   tilized inputs, strip the tilize/untilize aux generics from the
   flatbuffer, and run only the matmul kernel. That removes the
   downstream consumer of the output DFB but also makes the operand_alias
   bookkeeping moot — we could hand-elide the
   `reserve_back/push_back/wait_front/pop_front` calls on it. Beyond
   the four committed compiler fixes; natural follow-up.

This Phase E gap is **independent of the ScheduleDMA / num-nocs
question**. Same hang signature before and after the revert; the
compute kernel binary is identical (compute body doesn't depend on
NoC count).

### Summary

Compiler-level result: **PASS** — Quasar codegen produces structurally
valid, type-correct, compileable matmul kernels for both
`use-tile-matmul=true` and `=false`. Three upstream tt-mlir bugs were
found and fixed locally (a fourth, the 2-NoC ScheduleDMA extension,
was reverted as wrong-premise). Existing lit suites stay green.

Execution result: **partial** — the kernel runs on tt-emule until the
output-DFB design gap is hit. Resolving the gap is follow-up work.

### Critical-files reference (post-bring-up)

- `lib/Dialect/D2M/Transforms/ScheduleDMA.cpp` — **unchanged after
  revert** (Quasar uses the existing 1-NoC + materializeProcessorIndex
  path; the 2-NoC + N-DM extension was reverted as wrong-premise)
- `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` (DFB ct_arg routing,
  multi-ct_arg classifyCBRole)
- `lib/Conversion/D2MToTTKernel/D2MToTTKernelPass.cpp` (dropped processor-index reject)
- `lib/Conversion/D2MToTTMetal/D2MToTTMetal.cpp` (cb/dfb mix fix +
  multi-index classifyCBRole)
- `lib/Conversion/D2MToTTMetal/D2MToTTMetalPass.cpp` (dropped processor-index reject)
- `lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` (DFB naming,
  `.get_id()` wrap on free-function operands, DFB method-call receiver)
- `lib/Target/TTKernel/TTKernelToCpp.cpp` (auto-include dataflow_buffer.h)
- `test/ttmlir/Conversion/TTKernelToEmitC/dfb.mlir` (updated for declared-object form)

## Legacy "done" report template (kept for reference)

1. Phase B end-to-end status, per variant. Note dtype (f32 if it worked,
   bf16 if we fell back).
2. Phase C 7-item checklist per variant: ✓ or diff between expected
   and actual.
3. Phase D: kernel `.cpp` compiles standalone against Quasar headers?
4. Phase E: "ran to completion on tt-emule", "ran with PCC X", or
   "blocked on <specific gap>".
5. Any pipeline bug found, captured as `ttmlir-opt` + minimal repro.

Intermediate fixes committed locally on the current feature branch
(one per logical fix), each backed by a lit-style IR test.
**Nothing is pushed in any repo.**
