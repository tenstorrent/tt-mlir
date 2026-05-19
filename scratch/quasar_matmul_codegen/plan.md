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
+ `enable-compute-thread-tiling=true` + f32 + 2048×2048×2048) is untested
by any existing test, so real bugs are expected. The fix loop (§Phase F)
runs inside B/C/D, not after.

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
   Pass `num-datamovement-processors=6` and `num-nocs=2` explicitly to
   avoid system-desc defaults polluting the schedule.
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

## "Done" report

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
