---
name: add-ttir-d2m-lowering
description: >-
  Elementwise TTIR→D2M→TTMetal path: tablegen, TTIRToD2M.cpp, D2MToTTKernel.cpp, and
  — only when the kernel API callee is new — TTKernelIncludesMap.h (per-op
  api/compute/eltwise_unary/*.h mapping for JIT). Does not edit
  D2MGenericRegionOps.cpp or TTKernelToCpp.cpp. Not for reductions, matmul, views, or CCL.
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

   If the TTKernel op takes **i32-encoded scalar params** (float attrs bit-reinterpreted, or int
   attrs, or a runtime scalar Value), reuse the shared helpers defined at the top of the anonymous
   namespace rather than re-inlining a lambda:

   - `floatAttrToI32Bits(rewriter, loc, attr)` — `FloatAttr` → i32 bits (e.g. `selu` scale/alpha,
     `clamp_scalar` float min/max).
   - `intAttrToI32(rewriter, loc, attr)` — `IntegerAttr` → sign-extended i32 (e.g. `clamp_scalar` int
     min/max).
   - `scalarToI32Bits(rewriter, loc, value)` — runtime scalar `Value` → i32 (float widened+bitcast,
     int sign-extended/truncated). Used by `binop_with_scalar`-style scalar rhs lowerings.

   Ops with scalar attributes typically need a dedicated `else if constexpr
   (std::is_same_v<SFPUOp, ttkernel::FooTileOp>)` branch in the `D2MSFPUOpsRewriter` body that pulls
   attrs off `op` and calls the shared helper — see the `SeluTileOp` / `ClampScalarTileOp` branches
   as templates.

4. **`include/ttmlir/Target/TTKernel/TTKernelIncludesMap.h`** *(only if the kernel API callee is new)* —
   the `ScopedModuleHelper` in `lib/Target/TTKernel/TTKernelToCpp.cpp` no longer hardcodes
   `api/compute/eltwise_unary/*.h`. It walks the region and looks up each `emitc.call_opaque`
   callee in `getCalleeToHeadersMap()`. If your op lowers to a tt-metal SFPU helper
   (`foo_tile` / `foo_tile_init`) that isn't already in that map, add entries like:

   ```cpp
   {"foo_tile",      {"api/compute/eltwise_unary/foo.h", ""}},
   {"foo_tile_init", {"api/compute/eltwise_unary/foo.h", ""}},
   ```

   The callee string must match the `TTKernel_SFPUOp<"foo_tile", …>` / `TTKernel_InitOp<"foo_tile_init">`
   name in `TTKernelOps.td` exactly. **Do not** edit `TTKernelToCpp.cpp` to add includes directly —
   the old unconditional `emitc::IncludeOp` block was removed. Without a map entry, wormhole JIT can
   fail with "`foo_tile` was not declared in this scope" in `chlkc_unpack.cpp`.

**Out of scope here:** `D2MGenericRegionOps.cpp`, `TTKernelToCpp.cpp`. For TTNN / flatbuffer / full
builder parity across all targets, use `.claude/skills/add-op/SKILL.md`.

**Tests (minimal):** extend **existing** TTIR→D2M lit at
`test/ttmlir/Conversion/TTIRToD2M/named_to_generic.mlir`. Chain the new op into the SSA dataflow of
the existing `named_elementwise` function (bump the `%N` numbering and add a
`// CHECK: d2m.tile_<op>` + the `ttir.<op>` call) — **do not** create a separate
`named_elementwise_*` func for the new op. **No** lit under `test/ttmlir/Conversion/D2MToTTKernel/`
is required.

**Golden (TTMetal-only, no TTNN):** add `ttir_<op>.mlir` under `mlir_snippets/ttir/` — **one
snippet per new op** so `test_parse_split_ops.py` exercises parse/split for each. Add the golden
in `tools/golden/mapping.py` and the matching `@tag` / `@parse` / `@split` in
`tools/builder/ttir/ttir_builder.py` (same pattern as `square` / `exp`: pass `output_type_mlir`
into the golden, no **`_op_proxy`**).

For ops that carry MLIR attributes (e.g. SELU's `scale` / `alpha`, clamp's `min` / `max`), the
golden function should accept the **MLIR attr types** (`FloatAttr`, `IntegerAttr`, …) as
positional arguments and unpack them internally with `unpack_mlir_attr` — do **not** give the
golden Python-level defaults that duplicate the tablegen `DefaultValuedAttr`. The builder
`@tag` method is allowed to keep Python-float defaults as a caller convenience; just convert them
to `FloatAttr.get_f32(...)` and pass the `FloatAttr` directly into the golden (both from `@tag`
and from `@parse`, where you already have the attr off `old_op`). Mirror the
`ttnn_clamp_scalar_golden` / `ttnn_leaky_relu_golden` shape for this.

In `test/python/golden/ttir_ops/eltwise/test_ttir_unary.py` (or sibling), mark the op with
`SkipIf("ttnn", "emitc", "emitpy", "sim")` so it runs only on `ttmetal` on silicon until TTNN
lowering exists. `SkipIf` is already imported from `test_utils`; prefer it over the more verbose
`Marks(pytest.mark.skip_config([...]), …)` form.

Run `cmake --build build` after changes.

## Checklist

- [ ] `D2M_Tile*` in `D2MGenericRegionOps.td` (tablegen only; no extra `.cpp` for D2M tile op)
- [ ] `D2MNamedElementwiseRewriter<ttir::…, d2m::Tile…>` in the elementwise section of `populateTTIRToD2MPatterns`’s `patterns.add<{…}>`
- [ ] D2M→TTKernel map + rewriter in `D2MToTTKernel.cpp` (reuse `floatAttrToI32Bits` / `intAttrToI32` / `scalarToI32Bits` for any i32-encoded scalar params; don't inline new lambdas)
- [ ] `TTKernelIncludesMap.h`: entries for any new `*_tile` / `*_tile_init` callees (skip if the callee is already mapped). Do **not** touch `TTKernelToCpp.cpp`.
- [ ] Lit: chain the new op into the existing `named_elementwise` func in `named_to_generic.mlir` (no new func). No D2MToTTKernel lit required.
- [ ] Golden: one `mlir_snippets/ttir/ttir_<op>.mlir` **per new op** + `mapping.py` golden (take `FloatAttr`/`IntegerAttr` positionally and `unpack_mlir_attr` inside for ops with attrs — no Python defaults) + `ttir_builder.py` `@tag`/`@parse`/`@split` + `SkipIf("ttnn", "emitc", "emitpy", "sim")` for ttmetal-only-on-silicon (no TTNN)
