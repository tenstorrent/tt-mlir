# Chisel V2 \- Testing Plan

## Section 1 \- Overview

This document describes the testing strategy for Chisel V2. The primary constraint is that **CI has no TT hardware**, so all tests that run in CI must mock or stub the TTRT runtime layer. Tests requiring silicon are gated behind a dedicated marker and run only on hardware CI.

### 1.1 Test Categories

| Category | Dependencies | CI | Description |
| :---- | :---- | :---- | :---- |
| Unit | PyTorch only | Yes | Pure logic: metrics, config, report, tensor pool |
| MLIR Parsing | ttmlir bindings | Yes | Module parsing, op/SSA/attribute extraction |
| Registry | ttmlir bindings | Yes | Single-module TTNN op tracking |
| Golden Executor | ttmlir \+ tools/golden | Yes | CPU golden correctness per op |
| Callback Integration | Mocked TTRT | Yes | Full callback flow with mock runtime |
| End-to-End | Mocked TTRT \+ real MLIR | Yes | bind() through CSV report |
| Silicon | Real TTRT \+ device | No | Device tensor read-back, multi-chip |

### 1.2 Pytest Markers

```ini
[tool:pytest]
markers =
    unit: Pure logic tests, no external dependencies beyond PyTorch
    mlir: Requires ttmlir Python bindings
    integration: Mocked runtime integration tests
    e2e: End-to-end with mocked runtime and real MLIR fixtures
    silicon: Requires TT hardware, skipped in software-only CI
```

### 1.3 CI Target

```
pytest tools/chisel/tests/ -m "not silicon" --tb=short
```

## Section 2 \- Test Infrastructure

### 2.1 Fixtures (`conftest.py`)

```py
@pytest.fixture
def ttnn_mlir_string():
    """Minimal TTNN MLIR module as inline string (no file I/O)."""

@pytest.fixture
def mock_runtime():
    """Mocked DebugHooks + opContext + programContext + binary.
    Simulates TTRT callback dispatch without hardware."""

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for reports and cached tensors."""

@pytest.fixture
def chisel_config(tmp_path):
    """Writes a test config.yaml to tmp_path, returns path."""
```

### 2.2 Mock Runtime

The mock runtime replaces TTRT types so that callback tests run without a device:

- **MockBinary**: Holds a TTNN MLIR string in `.mlir.source`.
- **MockProgramContext**: Tracks a tensor pool as a plain dict.
- **MockOpContext**: Yields op location, debug string, and input/output tensor refs.
- **MockDebugHooks**: Captures registered callbacks and drives the preop→\[hw\]→postop dispatch sequence.

## Section 3 \- Unit Tests

### 3.1 Metrics (`test_metrics.py`)

Tests for the shared `tools/golden/metrics.py` module.

| Test | Description |
| :---- | :---- |
| `test_pcc_identical_tensors` | PCC of identical tensors \= 1.0 |
| `test_pcc_uncorrelated` | PCC of random uncorrelated tensors ≈ 0.0 |
| `test_pcc_negatively_correlated` | PCC of negated tensor ≈ \-1.0 |
| `test_pcc_all_nan` | Both all-NaN → 1.0 |
| `test_pcc_all_inf_equal` | Both all-+Inf → 1.0 |
| `test_pcc_all_inf_mismatch` | \+Inf vs \-Inf → 0.0 |
| `test_pcc_single_element` | Scalar tensors use equality fallback |
| `test_pcc_constant_vector` | Zero-variance → equality fallback |
| `test_pcc_shape_mismatch_broadcastable` | e.g. \[1,64\] vs \[32,64\] succeeds |
| `test_pcc_shape_mismatch_incompatible` | Returns \-1 |
| `test_abs_err_zero` | Identical tensors → 0.0 |
| `test_abs_err_known` | \[1,2,3\] vs \[1,2,5\] → 2.0 |
| `test_abs_err_nan_nan` | Both NaN → 0.0 |
| `test_rel_err_zero` | Identical tensors → 0.0 |
| `test_rel_err_known` | Verify against hand-computed value |
| `test_to_common_shape_squeeze` | Verifies squeeze path |
| `test_to_common_shape_permute` | Verifies permutation path |
| `test_to_common_shape_flatten` | Verifies flatten fallback |

### 3.2 Config (`test_config.py`)

| Test | Description |
| :---- | :---- |
| `test_load_default_config` | Loads from `~/.config/chisel/config.yaml` |
| `test_load_custom_path` | `CHISEL_CONFIG` env var override |
| `test_config_missing_file` | Sensible defaults when file absent |
| `test_config_output_dir` | `output_dir` propagates correctly |
| `test_config_skip_op_regex` | Regex list parsed correctly |
| `test_config_invalid_yaml` | Raises clear error |

### 3.3 Report Writer (`test_report.py`)

| Test | Description |
| :---- | :---- |
| `test_report_creates_file_with_header` | Header matches spec: `ssa_value, ttnn_op, op_debug_string, input_shapes, output_shapes, pcc, atol, rtol` |
| `test_report_write_row` | Single row written correctly |
| `test_report_multiple_rows` | Row ordering preserved |
| `test_report_special_characters` | Commas/quotes in `op_debug_string` are CSV-escaped |

### 3.4 Tensor Pool (`test_tensor_pool.py`)

| Test | Description |
| :---- | :---- |
| `test_pool_store_retrieve` | Basic dict get/set |
| `test_pool_caching_saves_to_disk` | `caching=True` writes `.pt` files to `output_dir` |
| `test_pool_no_cache_no_files` | `caching=False` writes nothing |
| `test_tensor_value_set_execution_data` | Default copies `.data`; explicit arg overrides |
| `test_pool_preserves_across_programs` | Golden tensors survive pool clear (cross-program sharing) |

## Section 4 \- MLIR Parsing Tests (`test_mlir_parsing.py`)

These validate the "MLIR module from flatbuffer" path (ChiselV2 Section 3.2). Tests use inline TTNN MLIR strings as fixtures so no compilation is needed.

Example fixture:

```mlir
module {
  func.func @forward(%arg0: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
    %0 = "ttnn.relu"(%arg0) : (tensor<32x128xbf16>) -> tensor<32x128xbf16> loc(#loc1)
    return %0 : tensor<32x128xbf16>
  }
}
```

| Test | Description |
| :---- | :---- |
| `test_parse_ttnn_module_from_string` | `Module.parse(src, ctx)` produces a valid module |
| `test_extract_ops_from_module` | Walk module, verify op count and op names |
| `test_extract_ssa_values` | SSA result names (`%0`, `%1`, …) correctly extracted |
| `test_extract_op_attributes` | Attributes (e.g. `transpose_b=true`) are readable |
| `test_extract_input_output_shapes` | Tensor shapes extracted from op types |
| `test_hash_location` | `hash_location` returns consistent `(line, col)` tuples |
| `test_unknown_location_sentinel` | Ops without location info → `(-1, -1)` |

## Section 5 \- Registry Tests (`test_registry.py`)

Tests for the single-module TTNN registry (V2 removes TTIR/TTNN correlation).

| Test | Description |
| :---- | :---- |
| `test_registry_loads_all_ops` | All ops from parsed module are registered |
| `test_registry_lookup_by_location` | `find_op(location)` returns correct op |
| `test_registry_op_ordering` | Ops iterated in program order |
| `test_registry_multi_function` | Module with 2 `func.func`s, both tracked |
| `test_registry_tracks_tensor_locations` | Input/output SSA values linked to ops |
| `test_registry_skip_non_ttnn_ops` | `func.return`, `ttnn.empty` etc. excluded from comparison |

## Section 6 \- Golden Executor Tests (`test_golden_executor.py`)

Parameterized tests verifying CPU golden correctness for individual TTNN ops via `GOLDEN_MAPPINGS`.

| Test | Op | Input | Expected |
| :---- | :---- | :---- | :---- |
| `test_golden_relu` | `ttnn.relu` | `[-1, 0, 1, 2]` | `[0, 0, 1, 2]` |
| `test_golden_add` | `ttnn.add` | `[1,2]+[3,4]` | `[4,6]` |
| `test_golden_matmul` | `ttnn.matmul` | 2x3 @ 3x2 | Shape 2x2, values correct |
| `test_golden_softmax` | `ttnn.softmax` | Known input | Sum-to-1 per row |
| `test_golden_exp` | `ttnn.exp` | `[0, 1]` | `[1, e]` |
| `test_golden_missing_op` | (unlisted op) | — | Raises `ValueError` |
| `test_golden_preserves_dtype` | any | bf16 input | bf16 output (or expected upcast) |
| `test_golden_executor_populates_pool` | any | — | Output tensor present in `TensorPool` after execution |

## Section 7 \- Callback Integration Tests (`test_callbacks.py`)

Tests the full callback flow using the mock runtime (Section 2.2). No hardware required.

| Test | Description |
| :---- | :---- |
| `test_preop_initializes_module_on_first_call` | First preop call parses MLIR and builds registry |
| `test_preop_captures_input_tensors` | Input tensors copied to golden pool |
| `test_postop_executes_golden_and_compares` | Golden runs, PCC/atol/rtol computed, CSV row written |
| `test_postop_op_skipping` | Op matching `skip_op_regex` → device output replaced with golden |
| `test_callback_sequence` | preop→\[hw\]→postop ordering, state transitions verified |
| `test_preprogram_callback` | Initializes registry, starts new report section |
| `test_postprogram_callback` | Flushes report, clears device pool, preserves golden pool |
| `test_multi_program_golden_sharing` | Golden tensors from program 0 visible in program 1 preop |
| `test_multiple_callbacks_coexist` | Chisel \+ another callback both fire (multi-callback support) |

## Section 8 \- End-to-End Tests (`test_e2e.py`)

Uses mock runtime with real MLIR fixtures from `test/mlir/test_*.mlir`.

| Test | Description |
| :---- | :---- |
| `test_bind_and_run_add` | `chisel.bind()` \+ simulated add graph → CSV with 1 row, PCC ≈ 1.0 |
| `test_bind_and_run_matmul_softmax` | Multi-op graph → CSV with correct row count |
| `test_skip_op_produces_golden_output` | Skip softmax, verify downstream sees golden values |
| `test_report_output_matches_spec` | CSV columns match ChiselV2 Section 2.1 spec |
| `test_output_dir_created` | Report lands in configured `output_dir` |
| `test_e2e_all_ops` (parameterized) | `@pytest.mark.parametrize("mlir_file", glob("test/mlir/test_*.mlir"))` |

## Section 9 \- Silicon Tests (Hardware CI Only)

These tests are gated behind `@pytest.mark.silicon` and skipped in software-only CI.

| Test | Description |
| :---- | :---- |
| `test_debug_hooks_registration` | Real `DebugHooks.get()` registration with TTRT runtime |
| `test_device_tensor_readback` | `retrieve_tensor_from_pool` returns correct data from device |
| `test_multi_chip_comparison` | Tensor comparison across multiple chips |
| `test_op_skip_updates_device_pool` | `update_tensor_in_pool` replaces device tensor with golden |
