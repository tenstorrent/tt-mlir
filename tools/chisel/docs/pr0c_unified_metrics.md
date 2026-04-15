# PR 0c: Unified Metrics

## Goal

Consolidate 3 duplicate PCC/tensor comparison implementations into a single
`tools/golden/metrics.py` module. Builder, chisel, and ttrt all import from
this one location instead of maintaining separate copies.

## Background

Three separate implementations exist today:

| Location | Implementation | Used By |
|----------|---------------|---------|
| `tools/builder/base/builder_runtime.py` (lines 188-375) | numpy-based `get_atol_rtol_pcc()`, `check_outputs()` with full metrics dict | builder's `execute_fb()`, golden test infra |
| `tools/ttrt/common/util.py` (lines 72-179) | Near-identical copy with logging param + message string return | `callback.py`, `emitpy.py`, `run.py` |
| `runtime/tools/chisel/chisel/utils/metrics.py` (lines 7-156) | Pure torch `compute_pcc()` with `_to_common_shape()` | Old chisel `context.py` |

All three compute the same thing with minor differences in edge-case handling,
dependencies, and return types.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/golden/metrics.py` | Unified comparison module — PCC, atol, rtol, full metrics dict |
| `test/python/golden/test_metrics.py` | Unit tests for all metrics functions and edge cases |

### Modified Files

| File | Change |
|------|--------|
| `tools/golden/CMakeLists.txt` | Add `metrics.py` to `GoldenSources` |
| `tools/golden/__init__.py` | Export new public names from `.metrics` |
| `tools/builder/base/builder_runtime.py` | Remove `mask_torch_inf_nan`, `get_atol_rtol_pcc`; `check_outputs` delegates to `golden.metrics.compute_metrics()` |
| `tools/ttrt/common/util.py` | Remove `mask_torch_inf_nan`, `get_atol_rtol_pcc`, `get_topk_diff` |
| `tools/ttrt/common/callback.py` | Import from `golden.metrics` |
| `tools/ttrt/common/emitpy.py` | Import from `golden.metrics` |
| `tools/ttrt/common/run.py` | Import from `golden.metrics` |

## Implementation Details

### `tools/golden/metrics.py` — Canonical API

```python
from typing import Dict, List, Optional, Tuple
import torch

def mask_inf_nan(tensor: torch.Tensor) -> torch.Tensor:
    """Clone tensor and zero out inf/nan values. Returns cleaned clone."""

def compute_pcc(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> float:
    """Pearson correlation coefficient. Pure torch, no numpy.
    Handles: empty, all-nan, constant, single-element tensors."""

def compute_atol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max absolute difference: max(|golden - calculated|)."""

def compute_rtol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max relative difference: max(|(golden - calculated) / calculated|)."""

def compute_metrics(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Dict[str, float]:
    """Full comparison dict — superset of all existing result formats.

    Returns:
        {
            "pcc": float,
            "atol": float,
            "rtol": float,
            "allclose": bool,
            "max": float,
            "mean_absolute_error": float,
            "root_mean_square_error": float,
            "cosine_similarity": float,
        }
    """

def get_topk_diff(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    top_k: int,
    relative: bool = False,
) -> List[Tuple]:
    """Top-k absolute or relative differences.
    Returns: [(v_golden, v_calculated, diff, index), ...]
    Ported from ttrt's get_topk_diff."""
```

### Key Design Decisions

1. **Pure torch, no numpy** — Chisel's `compute_pcc` already proves this works.
   The numpy `np.ma.corrcoef` approach adds a dependency and requires
   `.detach().numpy()` conversions. The torch implementation uses
   `sum(x_centered * y_centered) / sqrt(sum(x²) * sum(y²))` directly.

2. **No logging parameter** — The ttrt variant passes `logging` only to emit
   debug messages inside PCC. Callers should log before/after instead.

3. **No message string in return** — The ttrt 4-tuple return
   `(atol, rtol, pcc, message_string)` couples formatting with computation.
   Callers format the returned dict however they like.

4. **`compute_metrics` returns a dict** — Superset of builder's `check_outputs`
   result dict and ttrt's callback result dict. Callers pick what they need.

### Edge Case Reconciliation

The three implementations disagree in a few spots. The unified version resolves:

| Edge case | builder/ttrt | chisel | Unified |
|---|---|---|---|
| Single-element tensor | `cosine_similarity` fallback + `isclose` | `allclose` fallback | `torch.isclose` (builder/ttrt approach, more precise for scalars) |
| Constant tensors (all same value) | `isclose` on max values | Zero-variance → `equal()` | Zero-variance check → `torch.equal` fallback |
| numpy vs torch PCC | `np.ma.corrcoef` | `sum(x*y) / sqrt(sx² * sy²)` | Pure torch (equivalent, no numpy dependency) |
| Shape mismatch | Caller's responsibility | `_to_common_shape` with squeeze/broadcast/permute/flatten | Not needed — single-module TTNN means shapes match by construction |
| bfloat16 upcast | Explicit `bfloat16 → float32` check | Generic `.to(float32)` | Generic `.to(float32)` for all low-precision dtypes |
| MaskedConstant return | Returns 1.0 | N/A (no numpy) | N/A — zero-variance check handles this |

### Migration Details

**`tools/builder/base/builder_runtime.py`:**
- Remove `mask_torch_inf_nan()` (line 188), `get_atol_rtol_pcc()` (line 198)
- `check_outputs()` (line 287) keeps its existing signature for backward
  compatibility. Internally calls `golden.metrics.compute_metrics()` and
  applies the pass/fail threshold logic + `TTBuilderGoldenException` on top.
- `convert_golden_intermediates_to_torch()` and
  `convert_golden_input_output_to_torch()` are unchanged — they handle
  `GoldenMapTensor → torch.Tensor` conversion, not comparison.

**`tools/ttrt/common/util.py`:**
- Remove `mask_torch_inf_nan()` (line 72), `get_atol_rtol_pcc()` (line 84),
  `get_topk_diff()` (line 184)
- Add thin wrappers if the standalone ttrt wheel import path is a concern:
  ```python
  def get_atol_rtol_pcc(golden, calculated, atol, rtol, logging=None):
      from golden.metrics import compute_pcc, compute_atol, compute_rtol
      cal_atol = compute_atol(golden, calculated)
      cal_rtol = compute_rtol(golden, calculated)
      cal_pcc = compute_pcc(golden, calculated, atol=atol, rtol=rtol)
      msg = f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}"
      return (cal_atol, cal_rtol, cal_pcc, msg)
  ```
  Or update the 3 call sites directly if standalone use is not a concern.

**`tools/ttrt/common/callback.py`:**
- Replace duplicated results-dict construction (lines ~195-224) with
  `golden.metrics.compute_metrics()`.

### CMake Changes

**`tools/golden/CMakeLists.txt`:**
```cmake
declare_mlir_python_sources(GoldenSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    mapping.py
    metrics.py          # <-- new
)
```

**`tools/golden/__init__.py`:**
```python
from .mapping import *
from .metrics import (
    mask_inf_nan,
    compute_pcc,
    compute_atol,
    compute_rtol,
    compute_metrics,
    get_topk_diff,
)
```

## Porting Notes

### Implementation base: chisel's `compute_pcc` + builder edge cases

The unified `compute_pcc()` is based on chisel's pure-torch implementation
(`runtime/tools/chisel/chisel/utils/metrics.py:105-156`) because:
- No numpy dependency
- Cleaner edge-case handling with explicit masking
- Same mathematical result as numpy's `corrcoef`

Merge in builder/ttrt edge cases:
- Single-element `isclose` fallback (builder line 270-276)
- bfloat16 → float32 upcast (builder line 137-140)
- Empty tensor handling (builder/ttrt line 108-117)

### Old chisel (`runtime/tools/chisel/`)

Either replace `runtime/tools/chisel/chisel/utils/metrics.py` body with
re-exports from `golden.metrics`, or leave as-is if the old chisel is being
deprecated by the rewrite. The new `tools/chisel/` imports from
`golden.metrics` directly.

## Test Plan

### `test/python/golden/test_metrics.py`

**PCC tests:**
- `test_pcc_identical_tensors()` — PCC of identical tensors is 1.0
- `test_pcc_negated_tensors()` — PCC of negated tensors is -1.0
- `test_pcc_orthogonal_tensors()` — PCC of orthogonal tensors is ~0.0
- `test_pcc_single_element_close()` — single-element tensors within tolerance return 1.0
- `test_pcc_single_element_far()` — single-element tensors beyond tolerance return 0.0
- `test_pcc_constant_tensors_equal()` — constant tensors with same value return 1.0
- `test_pcc_constant_tensors_different()` — constant tensors with different values return 0.0
- `test_pcc_bfloat16()` — bfloat16 tensors are handled correctly

**Atol/rtol tests:**
- `test_atol_identical()` — atol of identical tensors is 0.0
- `test_atol_known_diff()` — `[1.0]` vs `[2.0]` yields 1.0
- `test_rtol_identical()` — rtol of identical tensors is 0.0

**Edge case tests:**
- `test_empty_tensors()` — empty tensors with same shape return 1.0
- `test_all_nan_tensors()` — both all-NaN returns 1.0
- `test_mixed_nan_tensors()` — one all-NaN returns 0.0
- `test_inf_tensors()` — inf handling
- `test_scalar_tensors()` — 0-dim tensors

**Full metrics tests:**
- `test_compute_metrics_returns_all_keys()` — verify all expected keys in result dict
- `test_compute_metrics_identical()` — pcc=1.0, atol=0.0, allclose=True

**Top-k tests:**
- `test_topk_diff_basic()` — verify correct top differences returned
- `test_topk_diff_relative()` — verify relative mode

**Test dependencies:** Only `torch` — no MLIR, no hardware, no numpy.

### Regression: existing tests

- `pytest test/python/golden/` — exercises builder's `check_outputs` via
  `execute_fb`. Must pass unchanged since `check_outputs` keeps its signature.
- ttrt tests — verify run/callback flows still work after import path changes.

## Dependencies

Included in Chisel PR 1 — `tools/golden/metrics.py` is created there directly.
This section documents the consolidation rationale but is no longer a separate
prerequisite PR.
