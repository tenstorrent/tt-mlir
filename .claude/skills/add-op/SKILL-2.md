---
name: add-op
description: >
  How to add a new operation (op) to the tt-mlir compiler across all layers: TTIR/TTNN dialect
  definitions, StableHLO composite conversion, TTIR-to-TTNN conversion, EmitC/EmitPy conversions,
  flatbuffer schema and serialization, runtime implementation, OpModel, ttir_builder, golden
  functions, and all associated tests. Use this skill whenever the user asks to add an op, implement
  an op, create a new operation, add support for a TTNN op, or mentions adding an op to the compiler
  pipeline. Also trigger when the user wants to know what files to change for a new op, or asks about
  the op-adding workflow.
---

# Adding an Op to tt-mlir

Adding an op touches ~15-30 files. Find the most similar existing op and follow its pattern.

**Implementation order:**
1. TTNN dialect definition → 2. TTIR dialect definition → 3. TTIR→TTNN conversion →
4. EmitC/EmitPy conversions → 5. StableHLO composite→TTIR (if needed) →
6. Flatbuffer schema + serialization → 7. Runtime → 8. OpModel → 9. TTIRBuilder, goldens, tests

**Key principle:** Start from the TTNN C++ API (`third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/`) and work outward. Consult `references/ttnn_type_mapping.md` for C++→MLIR type mappings.

**Before starting, identify:** the TTNN C++ API signature, a similar existing op as template, which args are optional (`AttrSizedOperandSegments`), naming conflicts, data type requirements, and tensor shape conventions.

## Step 1: TTNN Dialect Definition

**`include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`** — Model the TTNN C++ API closely. Map each parameter using `references/ttnn_type_mapping.md`. Add `AttrSizedOperandSegments` if there are optional operands, `TTNN_MemoryConfigOpInterface` if it takes memory_config. Set `let hasVerifier = 1`.

**`lib/Dialect/TTNN/IR/TTNNOps.cpp`** — Implement verifier for device-specific constraints.

## Step 2: TTIR Dialect Definition

**`include/ttmlir/Dialect/TTIR/IR/TTIROps.td`** — Simplified, device-agnostic version. Drop device-specific attrs (memory_config, compute_config, sub_device_id, topology). Base classes: `TTIR_NamedOp`, `TTIR_ElementwiseUnaryOp`/`BinaryOp`, or `TTIR_DPSOp`.

**`lib/Dialect/TTIR/IR/TTIROps.cpp`** — Implement verifier for mathematical semantics.

**Layout workarounds** (if metal kernel needs ROW_MAJOR for certain operands): add workaround in `TTNNWorkaroundsPass.h/.cpp` and `extraClassDeclaration` on the TTNN op. See `PagedScaledDotProductAttentionDecodeOp` for reference.

## Step 3: TTIR → TTNN Conversion

**`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`** — Add `OpConversionPattern<ttir::YourOp>`, register in `populateTTIRToTTNNPatterns`. Pass `nullptr` for TTNN-only attrs (memory_config, compute_config). For device refs use `::ttnn::utils::getOrInsertDevice(rewriter, op)`.

## Step 4: TTNN → EmitC

**`lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp`** — Use `EmitCTTNNEmitter`. Args must match TTNN C++ API order exactly. Register in `populateTTNNToEmitCPatterns`.

For `ttnn::experimental::` ops, override `getPrefixSearchPattern`/`getPrefixSwapPattern`. For multi-output ops returning `std::vector<Tensor>`, manually create `CallOpaqueOp` and extract via `SubscriptOp` (see `AllToAllDispatchOpConversionPattern`).

## Step 5: TTNN → EmitPy

**`lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp`** — Use `EmitPyTTNNEmitter` with keyword arg names. Register in `populateTTNNToEmitPyPatterns`. Pass `this->isGoldenModeEnabled()` to emitter.

**Important:** Keyword args must come AFTER positional args in the generated Python call, or you'll get syntax errors.

For `ttnn.experimental.` ops, override prefix methods. For GlobalSemaphore injection, see `DistributedRMSNormOpConversionPattern`.

## Step 6: StableHLO Composite → TTIR

**`lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`** — For simple 1:1 mappings: `patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::YourOp>>(context, "tenstorrent.your_op")`. For ops with optional operands or attribute conversion, write a custom pattern. Register in `populateStableHLOCompositeLegalizationPatterns`.

## Step 7: Flatbuffer Schema + Serialization

**`include/ttmlir/Target/TTNN/operations/<category>.fbs`** — Define table. Use `null` for absent optional refs. Multi-output ops use named output fields.

**`include/ttmlir/Target/TTNN/program.fbs`** — Add to `OpType` union (alphabetical).

**`lib/Target/TTNN/TTNNToFlatbuffer.cpp`** — Add `createOp` overload (use offset 0 for absent optional refs) and dispatch in `emitTTNNOperation`.

## Step 8: Runtime

| File | Action |
|------|--------|
| `runtime/lib/ttnn/operations/<category>/your_op.h` | New header with `run()` declaration |
| `runtime/lib/ttnn/operations/<category>/your_op.cpp` | New impl: get tensors from pool, call `::ttnn::your_op(...)`, insert result |
| `runtime/lib/ttnn/operations/CMakeLists.txt` | Add .cpp to `TTNN_OPS_SRCS` |
| `runtime/lib/ttnn/program_executor.cpp` | Add include + switch case dispatching to `operations::your_op::run()` |
| `runtime/lib/ttnn/runtime.cpp` | Add cases to `getOpOutputRef` and `getOpInputRefs` |
| `runtime/include/tt/runtime/detail/ttnn/ttnn.h` | Add TTNN library include |

For multi-output ops, insert each result separately into the tensor pool.

## Step 9: OpModel

**`include/ttmlir/OpModel/TTNN/MetalHeaders.h`** — Add TTNN metal header include.

**`include/ttmlir/OpModel/TTNN/TTNNOpModel.h`** — Add `OpModel<YourOp>` specialization. For elementwise ops, inherit from `UnaryEltwiseOpModel`/`BinaryEltwiseOpModel`.

**`lib/OpModel/TTNN/TTNNOpModel.cpp`** — Implement `getOpConstraints`/`getOpRuntime` (guarded by `#ifdef TTMLIR_ENABLE_OPMODEL`).

**`lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp`** — **REQUIRED for ALL ops.** Add at minimum stub implementations using `issueErrorForGetOpConstraints`/`issueErrorForGetOpRuntime` with `MissingMetalDefinition`.

## Step 10: TTIRBuilder

**`tools/builder/ttir/ttir_builder.py`** — Add three methods:
- **`@tag`** builder method — compute golden output, create op. Match MLIR attr types exactly (`SI32Attr` → `IntegerType.get_signed(32)`, not `get_signless`).
- **`@parse`** method — reconstruct from existing module via `global_dict`.
- **`@split`** method — create single-op Module.

For multi-output ops: return `Tuple[OpResult, ...]`, pass all result types to constructor, set golden for each result. See `sort()`, `max_pool2d_with_indices()`.

## Step 11: Golden Functions

**`tools/golden/mapping.py`** — Add golden implementations for TTIR and TTNN versions. Register in mapping dicts.

**Important:** Use `torch.mul()`/`torch.add()` instead of `*`/`+` operators (GoldenMapTensor doesn't support arithmetic operators). Parameter order must match builder call order: tensor inputs → attributes → `output_type_mlir`.

## Step 12: Precompiled Headers

**`tools/ttnn-standalone/ttnn-precompiled.hpp`** — Add TTNN operation header include.

## Step 13: Tests

| Test | File |
|------|------|
| TTIR→TTNN conversion | `test/ttmlir/Dialect/TTNN/<op>/simple_<op>.mlir` |
| Negative/verifier | `test/ttmlir/Dialect/TTNN/<op>/<op>_negative.mlir` (use `not ttmlir-opt`) |
| StableHLO composite | `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_<op>.mlir` |
| EmitC pipeline | `test/ttmlir/EmitC/TTNN/<op>/<op>.mlir` |
| Builder/golden | `test/python/golden/test_ttir_ops.py` (parametrize over shapes + targets `["ttnn", "emitpy", "emitc"]`) |
| OpModel | `test/unittests/OpModel/TTNN/Lib/TestOpModelLib.cpp` + `Op/TestOpModelInterface.cpp` |

For CCL ops, add `mesh-shape` pipeline parameter and test op folding for single-device.

## Verification

```bash
source env/activate
cmake --build build                           # Must pass
python .claude/skills/add-op/review/generate_review.py \
  --op-name <op_name> \
  --ttnn-test-dir test/ttmlir/Dialect/TTNN/<test_dir>/ \
  --emitc-test-dir test/ttmlir/EmitC/TTNN/<test_dir>/ \
  --pytest-filter <op_name> \
  --emitpy-input test/ttmlir/Dialect/TTNN/<test_dir>/simple_<op_name>.mlir \
  --emitc-input test/ttmlir/Dialect/TTNN/<test_dir>/simple_<op_name>.mlir
```

Review server at http://localhost:3118. If remote, forward port via VS Code Ports panel.

## Checklist

- [ ] TTNN tablegen + verifier (`TTNNOps.td`, `TTNNOps.cpp`)
- [ ] TTIR tablegen + verifier (`TTIROps.td`, `TTIROps.cpp`)
- [ ] TTIR → TTNN conversion (`TTIRToTTNN.cpp`)
- [ ] TTNN → EmitC conversion (`TTNNToEmitC.cpp`)
- [ ] TTNN → EmitPy conversion (`TTNNToEmitPy.cpp`)
- [ ] StableHLO composite → TTIR (`StableHLOLegalizeCompositePass.cpp`)
- [ ] Flatbuffer table + OpType union + serialization (`.fbs`, `program.fbs`, `TTNNToFlatbuffer.cpp`)
- [ ] Runtime header, impl, CMake, executor, runtime.cpp, ttnn.h
- [ ] OpModel: MetalHeaders.h, TTNNOpModel.h/.cpp, TTNNOpModelInterface.cpp (stubs at minimum)
- [ ] TTIRBuilder: builder, parser, split (`ttir_builder.py`)
- [ ] Golden functions (`mapping.py`)
- [ ] Precompiled headers (`ttnn-precompiled.hpp`)
- [ ] Tests: conversion, negative, StableHLO, EmitC, builder/golden, OpModel
