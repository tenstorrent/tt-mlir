# Chisel Failure Analysis

Generated: 2026-04-22

**Total failures across all runs:** 237 entries across 26 ops

---

## Priority by volume and severity

| # | Op | Count | Category | Severity |
|---|---|---|---|---|
| 1 | `ttnn.deallocate` | 88 | Shape/dtype tracking | High — broad upstream impact |
| 2 | `ttcore.load_cached` | 20 | MLIR metadata wrong | High — affects dependent ops |
| 3 | `ttnn.conv2d` | 32 | Harness bug | Medium — easy fix |
| 4 | `ttnn.slice_static` | 14 | Multiple bugs | High — 3 distinct issues |
| 5 | `ttnn.to_memory_config` | 14 | Multiple bugs | High — 3 distinct issues |
| 6 | `ttnn.typecast` | 11 | Shape expansion + broken u32 | High |
| 7 | `ttnn.softmax` | 6 | Numerical precision | Medium |
| 8 | `ttnn.constant` | 5 | bf16 materialization | High — zero PCC |
| 9 | `ttnn.scatter` | 7 | Output shape bug | Medium |
| 10 | `ttnn.empty` | 7 | Uninitialized output | Medium |
| 11 | `ttnn.embedding_bw` | 4 | Shape expansion + harness | Medium |
| 12 | `ttnn.multiply` | 4 | Broadcast collapse | Medium |
| 13 | `ttnn.paged_update_cache` | 4 | Checker slot misattribution | Low — harness bug |
| 14 | `ttnn.linear` | 3 | block_sharded untilize | Medium |
| 15 | `ttnn.relu` | 2 | block_sharded untilize | Medium |
| 16 | `ttnn.add` | 3 | Broadcast collapse | Medium |
| 17 | `ttnn.conv3d` | 3 | Harness bug | Low |
| 18 | `ttnn.prepare_conv2d_weights` | 3 | Weight layout transform | High — zero PCC |
| 19 | `ttnn.reshape` | 2 | Checker logic inverted | Low — harness bug |
| 20 | `ttnn.conv_transpose2d` | 2 | Harness bug | Low |
| 21 | `ttnn.fill_cache` | 2 | Test data size mismatch | Low — test bug |
| 22 | `ttnn.embedding` | 1 | Tile coordinate mapping | Medium |
| 23 | `ttnn.subtract` | 1 | Sign inversion / bf16 bug | High — negative PCC |
| 24 | `ttnn.clamp_tensor` | 1 | Harness bug | Low |
| 25 | `ttnn.batch_norm_inference` | 1 | Missing golden function | Low |
| 26 | `ttnn.where` | 1 | Missing golden function | Low |

---

## Category 1: Test Harness Bugs (Golden Executor KeyError)

Operand not found in the inputs dict at `executor.py:50`. The checker isn't wiring
all inputs before calling the golden function. These are **not op bugs**.

| Op | Failing operand | Notes |
|---|---|---|
| `ttnn.conv2d` | `%arg1`, `%arg3` | Weight/bias not materialized; 22 of 32 entries |
| `ttnn.conv3d` | `%4` | All 3 entries, identical error |
| `ttnn.conv_transpose2d` | `%3` | Weights in system_memory not fetched |
| `ttnn.clamp_tensor` | `%arg1`, `%arg2` | Only `%arg0` mapped; min/max tensors missing |
| `ttnn.embedding_bw` | `%3`, `%6` | Embedding table not passed; 2 of 4 entries |
| `ttnn.reshape` | — | Checker compares output against input shape instead of declared target shape |
| `ttnn.paged_update_cache` | — | Checker attributes si32 indices tensor to `%arg0` slot instead of bf16 cache tensor |

**Fix:** Audit `executor.py` input-gathering logic to ensure all operands (not just `%arg0`) are
materialized before the golden call. Special-case system_memory tensors.

---

## Category 2: Numerical Failures (pcc_fail)

### `ttnn.softmax` — 6 failures, PCC 0.18–0.91

- All failures on `golden_vs_runtime_tensor`
- f32 inputs consistently worse (PCC 0.18–0.51) than bf16 (PCC 0.91)
- Every failing case has `numericStable = false`
- `math_fidelity = hifi4` and `fp32_dest_acc_en = true` still produce poor results

**Root cause:** f32 reduction accumulates error in exp/normalize without the stable log-sum-exp
algorithm. BF16's lower range accidentally masks the issue.

**Fix:** Enable `numericStable = true`, or investigate the f32 softmax reduction kernel for
accumulated rounding error.

---

### `ttnn.constant` — 5 failures, PCC = 0.0 (all)

- All failures on `golden_vs_runtime_tensor` with bf16 dtype
- Both `dense_resource<dense_attr>` (external ref) and inline `dense<...>` fail
- Consistent across all memory layouts (row_major/system_memory, tile/DRAM, tile/L1)

**Root cause:** bf16 constant materialization is completely broken — zero PCC indicates no data
is reaching the output tensor. Likely the bf16 const data copy to device or tile layout
conversion is silently dropping/zeroing values.

---

### `ttnn.empty` — 7 failures, PCC = 0.0 (all)

- All `golden_vs_runtime_tensor`, f32 dtype, block_sharded L1 layout
- All shapes 32×32 or 128×128

**Root cause:** Tensor is allocated but not initialized. Golden reference expects specific
values; runtime returns zeroed/garbage memory. The initialization kernel may be missing or
not triggered for block_sharded L1 allocations.

---

### `ttnn.prepare_conv2d_weights` — 3 failures, PCC ≈ 0.0

- All `golden_vs_runtime_tensor`, bf16, kernels 3×3 and 7×7
- Failures across DRAM-interleaved and L1-height_sharded input configs (all fail)
- PCC range: -0.0005 to 0.031 — effectively random output

**Root cause:** OIHW→tile weight layout transformation produces corrupt output. Both sharded
and non-sharded configs fail, so the bug is in the core layout transform, not the memory config.

---

### `ttnn.embedding` — 1 failure, PCC = -0.158

- `golden_vs_runtime_tensor`, ui32 indices + bf16 embedding table
- Output: 2×4×10 bf16

**Root cause:** Tile coordinate mapping in the output layout is incorrect. The stride formula
`(d0*32 + d1, d2)` in the output tensor suggests a layout transformation bug in TTMetal codegen,
not a numerical precision issue.

---

### `ttnn.subtract` — 1 failure, PCC = -0.033

- `golden_vs_runtime_tensor`, 32×32 bf16, identical input/output layouts

**Root cause:** Negative PCC indicates sign inversion or complete output corruption, not
precision loss. Likely a bf16 arithmetic bug or wrong-sign kernel for the subtract operation.

---

## Category 3: Shape and Broadcasting Bugs

### `ttnn.add` and `ttnn.multiply` — broadcast collapse

Both ops fail with shape_mismatch when one operand is a small constant broadcast against a
large tensor:

- `add`: `[1,32,64,512] + [1,1,1,1]` → output is `[1,1,1,1]` (should be `[1,32,64,512]`)
- `multiply`: `[1,1,32,128] × [1,1,1,128]` → dim-2 collapses; `× [1,1,32,1]` → dim-3 collapses

**Root cause:** Broadcasting logic doesn't expand small operand shapes to match the output;
the runtime kernel appears to inherit the smallest operand's shape instead.

---

### `ttnn.scatter` — 7 failures

- 3 entries skipped due to `scatter_reduce_type = invalid` (expected)
- 4 `shape_mismatch` entries: output shape is indices tensor shape instead of input tensor shape
  - Example: input `[2272]`, indices `[256]` → output `[256]` (should be `[2272]`)
  - All 1D cases, `dim = 0`

**Root cause:** Output shape inference uses indices tensor shape rather than input tensor shape
for 1D scatter on dim 0.

---

### `ttnn.typecast` — 11 failures, two distinct bugs

**Bug 1 — Spurious batch dimensions (bf16→f32, 6 shape_mismatch):**
- Runtime returns `[1,1,N,M]` where MLIR declares `[N,M]`
- Tile-based memory layout (`{1x1}`, 16×4 tile grid) being misinterpreted as actual tensor shape

**Bug 2 — u32 conversion completely broken (f32/bf16→u32, 5 pcc_fail):**
- PCC range: -0.228 to -0.127 (all negative, indicating complete corruption)
- atol values up to 4.3×10⁹ — output is garbage
- Likely a missing cast kernel; bitcast used instead of semantic type conversion

---

### `ttnn.embedding_bw` — 4 failures

- 2 `shape_mismatch`: runtime returns `[1,1,N,M]` instead of `[N,M]` (same batch-dim bug as typecast)
- 2 `error`: golden executor KeyError for embedding table operand

---

### `ttnn.slice_static` — 14 failures, three distinct bugs

**Bug 1 — Shape mismatch on tile-boundary slices (8 shape_mismatch):**
- 3D case: slice `71×4×2 → 71×4×1` but output reports wrong last dim
- 1D case: expected 256-element output, runtime returns 2272 (tile-padded size leaking into shape)

**Bug 2 — dtype corruption for integer tensors (5 dtype_mismatch):**
- si32 input tensors are returned as bf16 from the runtime

**Bug 3 — Zero-size output dimension (1 error):**
- Step-based slice where `end == begin` on last dimension → output size 0 → crashes
  `torch.frombuffer` with `count=-1`

---

### `ttnn.reshape` — 2 failures

- Checker compares output against the **input** shape instead of the declared **target** shape
- Both entries fail `mlir_vs_tensor_ref` and `mlir_vs_runtime_tensor` with shape_mismatch

**Root cause:** Test infrastructure bug — verification logic is inverted.

---

## Category 4: Memory Layout / Runtime Errors

### block_sharded untilize constraint — affects `ttnn.linear`, `ttnn.relu`, `ttnn.to_memory_config`

All three ops fail with:
```
UntilizeWithUnpadding: output memory config layout must be INTERLEAVED
for block sharded input but got TensorMemoryLayout::BLOCK_SHARDED
```

The compiler emits block_sharded output layout for these ops, but the host-transfer
(`toHost()`) path requires INTERLEAVED output. Same root cause, one fix needed.

| Op | Count | Note |
|---|---|---|
| `ttnn.linear` | 3 | All failures are this error |
| `ttnn.relu` | 2 | All failures are this error |
| `ttnn.to_memory_config` | 5 of 14 | Combined with other issues below |

---

### `ttnn.to_memory_config` — 14 failures (3 sub-issues)

Beyond the block_sharded untilize issue above:

1. **height_sharded shape undercomputation (5 shape_mismatch):**
   - MLIR declares `[1,1,32,512]`, runtime returns `[1,32,64,512]`
   - Virtual-to-physical dimension folding (e.g., `d0 floordiv 8`) is computed incorrectly

2. **L1 OOM (3 errors):**
   - Shard allocation doesn't account for program circular buffer overhead
   - Example: 281 KB free, 524 KB needed; circular buffer at L1 530–906 KB collides with tensor

---

## Category 5: MLIR Metadata / Shape Tracking

### `ttcore.load_cached` — 20 failures

- 15 `shape_mismatch`, 2 `dtype_mismatch`, 3 `skipped`
- MLIR IR declares wrong shapes for cached constants:
  - `tensor<1x1x1x128xbf16>` → actual runtime shape `[1,1,32,128]`
  - `tensor<1xsi32>` → actual `[1,1,32,512]` with dtype corruption
- No numerical failures — purely metadata errors

**Root cause:** Constant shape inference doesn't correctly handle non-tile-aligned dimensions
or multi-dim tensors. The MLIR shape annotation is wrong before execution begins.

---

### `ttnn.deallocate` — 88 failures (largest single op count)

- 58 `shape_mismatch`, 26 `dtype_mismatch`, 4 OOM errors
- All entries have `force = false`
- Bidirectional dtype corruption: si32↔bf16 mismatch in both directions
- The deallocate op itself is semantically trivial — the failures reflect upstream IR state

**Root cause:** Upstream ops (reshape, layout conversions, type casts) don't propagate shapes
and dtypes correctly into operand type signatures. By the time `deallocate` is checked, the
declared tensor type no longer matches runtime buffer state. This is a symptom of broader
shape/dtype tracking failures upstream.

---

### `ttnn.fill_cache` — 2 failures

- Both `shape_mismatch`: expected depth=64, actual depth=3 in cache tensor
- `batch_offset = 0` for both

**Root cause:** Test data generation bug. The cache tensor was created at 1/21st the required
size. Not an op implementation issue.

---

## Category 6: Missing Golden Functions

| Op | Status | Note |
|---|---|---|
| `ttnn.batch_norm_inference` | `skipped` | No golden function registered |
| `ttnn.where` | `skipped` | No golden for bf16 on non-tile-aligned shape (13×37) |
| `ttnn.scatter` (3 entries) | `skipped` | `scatter_reduce_type = invalid` — intentional skip |
