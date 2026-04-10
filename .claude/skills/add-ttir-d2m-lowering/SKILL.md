---
name: add-ttir-d2m-lowering
description: >-
  Elementwise TTIR→D2M→TTMetal path: tablegen, TTIRToD2M.cpp, D2MToTTKernel.cpp, and when needed
  TTKernelToCpp.cpp (per-op api/compute/eltwise_unary/*.h includes for JIT). Does not edit
  D2MGenericRegionOps.cpp. Not for reductions, matmul, views, or CCL.
---

# TTIR elementwise → D2M (TTMetal path)

**Allowed edits (these layers):**

1. **Tablegen** — e.g. `include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td` (and any other `.td` you
   already own for the op). Pick the same base class as the nearest op (unary:
   `D2M_GenericRegionComputeUnaryDstOp`; typical binary: `…FPUOrSFPUBinary`; ternary: `…TernaryDstOp`).
   Prefer ops that need **no** hand-written C++ in `D2MGenericRegionOps.cpp`; that file is **out of
   scope** for this workflow.

2. **`lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`** — in `populateTTIRToD2MPatterns`, add one line to the
   big `patterns.add< … >` list with the other elementwise rewriters, e.g.
   `D2MNamedElementwiseRewriter<ttir::YourOp, d2m::TileYourOp>,` (keep ordering consistent with
   neighbors). Use `notifyMatchFailure` inside patterns, not `emitOpError`.

3. **`lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`** — extend `ComputeOpMap` / `IntComputeOpMap` and
   the `patterns.add<…D2MSFPUOpsRewriter…>` list to match the nearest unary/binary tile op.

4. **`lib/Target/TTKernel/TTKernelToCpp.cpp`** — when the lowered TTKernel calls tt-metal SFPU helpers such
   as `foo_tile_init()` / `foo_tile()`, the generated C++ must include the matching header under
   `api/compute/eltwise_unary/` (same layout as in tt-metal’s `tt_metal/hw/inc/`). In
   `ScopedModuleHelper`’s **compute** branch, add `emitc::IncludeOp` for that path (see existing
   `exp.h`, `log1p.h`, etc.). Without this, wormhole JIT can fail with “`foo_tile` was not declared in
   this scope” in `chlkc_unpack.cpp`.

**Out of scope here:** `D2MGenericRegionOps.cpp`. For TTNN / flatbuffer / full builder parity across all
targets, use `.claude/skills/add-op/SKILL.md`.

**Tests (minimal):** extend **existing** TTIR→D2M lit (e.g.
`test/ttmlir/Conversion/TTIRToD2M/named_to_generic.mlir`). **No** lit under
`test/ttmlir/Conversion/D2MToTTKernel/` is required.

**Golden (TTMetal-only, no TTNN):** add `ttir_<op>.mlir` under `mlir_snippets/ttir/`, golden + `ttir_builder.py`
**`@tag` / `@parse` / `@split`** (same pattern as `square` / `exp`: pass `output_type_mlir` into the golden,
no **`_op_proxy`**). In `test_ttir_ops/eltwise/test_ttir_unary.py` (or sibling) mark the op with
`Marks(pytest.mark.skip_config(["ttnn"]), …, ["emitc"], ["emitpy"])` so it runs for **`ttmetal`** only
until TTNN lowering exists.

Run `./build_and_test.sh` or `cmake --build build` after changes.

## Checklist

- [ ] `D2M_Tile*` in `D2MGenericRegionOps.td` (tablegen only; no extra `.cpp` for D2M tile op)
- [ ] `D2MNamedElementwiseRewriter<ttir::…, d2m::Tile…>` in the elementwise section of `populateTTIRToD2MPatterns`’s `patterns.add<{…}>`
- [ ] D2M→TTKernel map + rewriter in `D2MToTTKernel.cpp`
- [ ] `TTKernelToCpp.cpp`: `emitc::IncludeOp` for any new `api/compute/eltwise_unary/*.h` needed by generated kernels
- [ ] Lit: extend existing TTIRToD2M tests only (no D2MToTTKernel lit required)
- [ ] Optional golden: snippet + builder + `skip_config` for ttmetal-only (no TTNN)
