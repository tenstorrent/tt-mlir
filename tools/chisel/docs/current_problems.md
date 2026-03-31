# Chisel Rewrite: Problems and Solutions

This document describes the problems that motivated the Chisel rewrite, how they
were previously addressed in the builder and old Chisel (v1), and how the new
Chisel design solves them. It also lists open questions for future work.

## Summary

The Chisel rewrite addresses seven core problems:

1. **TTIR/TTNN cross-dialect correlation** — Comparing across two different IR
  dialects requires complex op correlation, fusion mismatch handling, and
   problems with erase inverse ops.
2. **TTNN op golden coverage** — `GOLDEN_MAPPINGS` has only around 60/140 ttnn golden ops
3. **Multi-program support** — Old Chisel only supported single-program
  execution, but real workloads involve forward/backward passes and training
   loops.
4. **Multi-chip/multi-device support** — Device tensors are sharded across
  chips; golden comparison must happen per-device, but `TensorRef` Python
   bindings are opaque and key APIs don't support multi-device.
5. **Duplicate metrics implementations** — Three separate PCC/comparison
  implementations exist (builder, ttrt, old chisel).
6. **Multi-output op support** — `getOpOutputRef()` only returns a single
  output; multi-output ops like `SortOp` and `BatchNormTrainingOp` are
   invisible to callbacks.
7. **DebugHooks callback safety** — Callbacks are copied by value on every op
  invocation, causing segfaults when Python callables are used without the GIL.

## 1. TTIR/TTNN Cross-Dialect Correlation

### Problem

Old Chisel compared TTIR ops (golden) against TTNN ops (device). These are
different IR dialects with different operation granularity:

- A single TTIR op may lower to multiple TTNN ops (e.g., TTIR `matmul` to
TTNN `linear` + `reshape`).
- Some TTNN ops have no TTIR counterpart (layout ops, device management).
- Op fusion during lowering creates N:M mappings between TTIR and TTNN ops.

This required a complex `Registry` that correlated ops across the two modules
using MLIR source locations, handled fusion group mismatches with
`_merge_empty_golden_groups()`, and maintained separate `ExecutionType` enums
per module.

### How Old Chisel (v1) Handled It

- Maintained two `IRModule` instances (one TTIR, one TTNN) with separate
`execution_type` enums (`GOLDEN` vs `DEVICE`).
- Registry used source locations to correlate TTIR golden ops with TTNN device
ops.
- Required special handling for ops that existed in one dialect but not the other.
- `_merge_empty_golden_groups()` handled cases where TTIR ops had no direct
TTNN counterpart after fusion.

### Proposed Solution

New Chisel uses a **single TTNN module** for both golden and device execution.
The TTNN MLIR is extracted from the flatbuffer's `TTNNBinary.mlir.source` field
and parsed once on the first preop callback.

Benefits:

- **1:1 op correspondence** — no cross-dialect correlation needed.
- **No fusion mismatch handling** — both golden and device see the same ops.
- **Single Registry** tracking ops in one module, significantly simplified.
- **No `ExecutionType` enum** — removed entirely.

## 2. TTNN Op Golden Coverage

### Problem

`GOLDEN_MAPPINGS` in `tools/golden/mapping.py` historically contained only TTIR
and StableHLO op mappings. Since new Chisel compares at the TTNN level, it needs
golden implementations for TTNN operations.

### How the Builder Handles It

The builder only needs TTIR/StableHLO goldens because it computes goldens at
compile time before lowering to TTNN. The `get_golden_function()` lookup uses
TTIR op classes.

### Current State

TTNN golden functions are being added to `GOLDEN_MAPPINGS`. The mapping already
includes TTNN ops such as `ttnn.AbsOp`, `ttnn.SigmoidOp`, `ttnn.ReluOp`,
`ttnn.MatmulOp`, `ttnn.AddOp`, and others (~30+ ops).

### Remaining Work

A dedicated PR is needed to add golden implementations for the remaining TTNN
ops. Priority should be given to ops that appear most frequently in production
models.

## 3. Multi-Program Support

### Problem

Real workloads often involve multiple programs per binary (e.g., forward and
backward passes) and repeated execution of the same program (training loops).
Old Chisel only supported single-program execution.

### How Old Chisel (v1) Handled It

Old Chisel was a CLI tool that loaded a single flatbuffer, ran a single program,
and produced a single report. Multi-program execution required multiple
invocations with manual coordination.

### Proposed Solution

New Chisel supports multi-program execution with:

- **Program transition detection**: Tracks `Binary.id` (process-scoped monotonic
counter) and op counter to detect when a new program starts.
- **Asymmetric state reset**: On program transition, device tensor pool and op
index are cleared (tied to destroyed `ProgramContext`), while golden tensor
pool, IR module, registry, and executor are preserved.
- **Cross-program golden sharing**: Golden tensors from one program (e.g.,
shared weights, output-to-input chaining) are available to subsequent
programs without recomputation.
- **Per-program reporting**: Report writer groups ops by `program_index` in a
single CSV file.

## 4. Multi-Chip/Multi-Device Support

### Problem

Device tensors in multi-chip configurations are sharded across multiple chips.
Each chip holds a local shard of the full tensor, and golden comparison must
happen independently per device. This introduces several challenges:

1. **Per-device golden tensors**: The builder uses `GoldenMapTensor` with
   `shard_map: Dict[int, torch.Tensor]` keyed by logical device ID. Golden
   functions are applied per-shard via the `__torch_function__` protocol.
2. **Opaque Python bindings**: `TensorRef` is exposed to Python with zero
   property accessors — `shape`, `local_shape`, `mesh_shape`, and
   `shard_status` from the flatbuffer `TensorDesc` are not accessible.
3. **API limitation**: `retrieve_tensor_from_pool()` fatally errors if the
   tensor has multiple device shards (`hostTensors.size() != 1`). Only
   `get_op_output_tensor()` handles multi-device correctly, returning a
   `Dict[device_id, Tensor]`.

### How the Builder Handles It

- Golden tensors are `GoldenMapTensor` objects with one shard per device.
- The `golden()` callback iterates over `op_golden_tensor_map.items()` and
  compares each device's output independently.
- `get_op_output_tensor()` returns per-device tensors indexed by logical device
  ID (row-major order).

### Proposed Solution

New Chisel replicates the builder's per-device comparison model:
- Golden executor operates on `GoldenMapTensor` objects natively, applying
  golden functions per-shard.
- Postop comparison iterates over devices, computing per-device metrics.
- For tensor shape information, use `binary.get_program_ops_as_json()` as a
  workaround until `TensorRef` Python bindings are extended.

## 5. Duplicate Metrics Implementations

### Problem

Three separate PCC/tensor comparison implementations exist across the codebase:

| Location | Implementation | Used By |
|----------|---------------|---------|
| `tools/builder/base/builder_runtime.py` | numpy-based `get_atol_rtol_pcc()` with full metrics dict | builder's `execute_fb()` |
| `tools/ttrt/common/util.py` | Near-identical copy with logging param and message string return | ttrt callbacks |
| Old chisel `metrics.py` | Pure torch `compute_pcc()` with shape alignment | Old chisel `context.py` |

These implementations disagree on edge cases (single-element tensors, constant
tensors, bfloat16 handling), have different dependencies (numpy vs pure torch),
and return different types (dict vs tuple vs scalar).

### Proposed Solution

Consolidate into a single `tools/golden/metrics.py` module:
- **Pure torch, no numpy** — eliminates the numpy dependency and
  `.detach().numpy()` conversions.
- **`align_shapes()` as opt-in utility** — handles squeeze/broadcast/permute/
  flatten for minor layout differences; callers that already ensure matching
  shapes skip it.
- **`compute_metrics()` returns a dict** — superset of all existing result
  formats; callers pick what they need.
- Builder, ttrt, and chisel all import from this single module.

## 6. Multi-Output Op Support

### Problem

`getOpOutputRef()` returns `std::optional<TensorRef>`, supporting only
single-output operations. Multi-output ops return `std::nullopt` with a log
warning, making their outputs invisible to callbacks:

| Op | Outputs |
|----|---------|
| `SortOp` | sorted values, indices |
| `MaxPool2dWithIndicesOp` | pooled output, indices |
| `BatchNormTrainingOp` | output, mean, rstd |

This means Chisel cannot compare golden vs device for any multi-output op.

### Proposed Solution

Change `getOpOutputRef()` to return `std::vector<TensorRef>`:
- Single-output ops return a vector of size 1.
- Multi-output ops return a vector of size N.
- No-output ops (e.g., `DeallocateOp`) return an empty vector.

This is **not blocking** for initial Chisel integration — Chisel skips
multi-output ops without it. It extends coverage to additional ops when landed.

## 7. DebugHooks Callback Safety

### Problem

In `ProgramExecutor::execute()`, `getPreOperatorCallback()` and
`getPostOperatorCallback()` return `std::optional<CallbackFn>` **by value**,
copying the `std::function` on every op invocation. `runCallback()` also takes
`std::optional<CallbackFn>` by value, causing a second copy.

When the `std::function` wraps a `nb::callable` (nanobind Python callable),
each copy manipulates the Python object's reference count. If the calling thread
doesn't hold the GIL (e.g., in tt-xla), this causes a segfault.

### Proposed Solution

Eliminate all callback copies:
- **Return by const reference**: `getPreOperatorCallback()` and
  `getPostOperatorCallback()` return `const std::optional<CallbackFn> &`.
- **Accept by const reference**: `runCallback()` takes
  `const std::optional<CallbackFn> &`.
- **Move on registration**: `Hooks::get()` accepts callbacks by rvalue reference
  and moves them into storage.

This is a **blocking** fix — must land before Chisel registers Python callbacks
via `DebugHooks.get()`.

---

## Open Questions

### Where to bind Chisel in tt-xla?

tt-xla is a key execution path that currently has no golden comparison
capability. Chisel's callback-based design should make integration
straightforward, but the specific binding point needs to be determined.
Two approaches are under consideration:

**Option A: tt-xla compile options** — Expose an `enable_chisel` flag through
tt-xla's compilation options. tt-xla would create the `ChiselContext`, register
callbacks, and manage the lifecycle internally.

**Option B: Side-load PJRT plugin through Python ttrt bindings** — Register
Chisel's callbacks externally by calling `DebugHooks.get()` from Python before
tt-xla execution begins. Since `DebugHooks` is a global singleton, callbacks
registered from Python are visible to the C++ runtime regardless of who drives
execution. One potential problem does wheel works with it?

### Pre-program and post-program callbacks?

Current `DebugHooks` only supports preop/postop callbacks at the operation
level. Should we add program-level callbacks (`pre_program` / `post_program`)
for setup/teardown, reporting boundaries, or state management? This would
simplify Chisel's program transition detection logic.

---

## References

- [Architecture](architecture.md) — module structure, data flow, component details
- [Feature Overview](feature_overview.md) — capabilities, usage examples, multi-program design
- [Multi-Chip Analysis](multi_chip_analysis.md) — findings on multi-device support
- `tools/golden/mapping.py` — `GOLDEN_MAPPINGS` and `get_golden_function()`
- `tools/builder/base/builder_runtime.py` — builder's golden callback and `CallbackRuntimeConfig`
- `tools/builder/base/builder.py` — compile-time golden computation via `golden_map`

