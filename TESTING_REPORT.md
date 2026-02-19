# TT-MLIR Device Testing Report

This report identifies all tests in the tt-mlir repository that run on **actual Tenstorrent hardware devices** (silicon/hardware execution), as opposed to compiler-only tests.

## Summary

- **Total Device Test Files**: ~73 test files
- **Main Categories**:
  1. Python Golden Tests (52 files)
  2. TTNN-JIT Tests (21 files)
  3. PyKernel Demo Tests (~17 files)

---

## 1. Python Golden Tests (`test/python/golden/`)

**Total: 52 test files**

These tests compile MLIR dialects through the full compilation pipeline to flatbuffers and execute them on silicon devices. They use the `device` fixture from `conftest.py` which calls `tt_runtime.runtime.open_mesh_device()` to open actual hardware devices.

### Main Test Files (Root Level)

| Test File | Description |
|-----------|-------------|
| `test_metal_allocate.py` | Memory allocation tests for TTMetal backend |
| `test_metal_bfp8_typecast.py` | BFP8 type casting tests |
| `test_metal_dma.py` | DMA (Direct Memory Access) operations |
| `test_metal_layout.py` | Tensor layout transformations |
| `test_metal_masking.py` | Masking operations |
| `test_metal_matmul.py` | Matrix multiplication on TTMetal |
| `test_metal_reductions.py` | Reduction operations |
| `test_metal_tensor_collapsing.py` | Tensor shape collapsing |
| `test_metal_tilize.py` | Tile layout conversions |
| `test_metal_tms.py` | Tensor manipulation operations |
| `test_metal_virtual_grids.py` | Virtual grid configurations |
| `test_metal_virtual_grid_rowmajor.py` | Row-major virtual grids |
| `test_ttir_models.py` | Full TTIR model tests |
| `test_ttir_models_llama_tp.py` | Llama model with tensor parallelism |
| `test_ttir_ops.py` | General TTIR operations |
| `test_ttir_parallels.py` | Parallel execution tests |
| `test_ttir_eltwise_fusion.py` | Element-wise operation fusion |
| `test_ttir_fusing.py` | General fusion tests |
| `test_ttnn_ops.py` | General TTNN operations |
| `test_ttnn_fusing.py` | TTNN fusion tests |
| `test_stablehlo_ops.py` | StableHLO operation tests |
| `test_shardy_ops.py` | Shardy sharding operations (single chip) |
| `test_shardy_ops_n300.py` | Shardy operations for N300 (multi-chip) |
| `test_fabric_apis.py` | Multi-chip fabric communication tests |
| `test_composite_functions.py` | Composite function tests |
| `test_compute_kernel_config.py` | Compute kernel configuration |
| `test_rearrange.py` | Tensor rearrangement operations |
| `test_parse_split_ops.py` | Operation splitting tests |
| `test_utils.py` | Utility function tests |

### Subdirectory: `ttir_ops/`

TTIR (Tenstorrent Intermediate Representation) operation tests, organized by operation category:

#### `ttir_ops/eltwise/`
- `test_ttir_binary.py` - Binary element-wise operations (add, sub, mul, div, etc.)
- `test_ttir_unary.py` - Unary element-wise operations (exp, log, sqrt, etc.)
- `test_ttir_ternary.py` - Ternary operations (where, select, etc.)

#### `ttir_ops/convolution/`
- `test_conv2d.py` - 2D convolution operations
- `test_conv_transpose2d.py` - Transposed 2D convolution

#### `ttir_ops/matmul/`
- `test_matmul.py` - Matrix multiplication operations

#### `ttir_ops/normalization/`
- `test_normalization.py` - Normalization operations (LayerNorm, RMSNorm, etc.)

#### `ttir_ops/pooling/`
- `test_pooling.py` - Pooling operations (MaxPool, AvgPool, etc.)

#### `ttir_ops/reduction/`
- `test_reduction.py` - Reduction operations (sum, mean, max, min, etc.)

#### `ttir_ops/data_movement/`
- `test_data_movement.py` - Data movement operations (transpose, reshape, concat, etc.)

#### `ttir_ops/fusing/`
- `test_mish_fusing.py` - Mish activation fusion
- `test_rope_fusing.py` - RoPE (Rotary Position Embedding) fusion
- `test_sdpa_fusing.py` - Scaled Dot-Product Attention fusion
- `test_permute_matmul_fusing.py` - Permute-MatMul fusion
- `test_reshape_broadcast_reshape_to_repeat_fusing.py` - Complex reshape fusion patterns

#### `ttir_ops/workarounds/`
- `test_workarounds.py` - Tests for known issue workarounds

### Subdirectory: `ttnn_ops/`

TTNN (Tenstorrent Neural Network) operation tests:

#### `ttnn_ops/eltwise/`
- `test_ttnn_binary.py` - Binary operations in TTNN
- `test_ttnn_unary.py` - Unary operations in TTNN
- `test_ttnn_ternary.py` - Ternary operations in TTNN

### Subdirectory: `optimizer/`

**Note**: These tests require the `--require-opmodel` flag to run.

- `test_conv_sharding.py` - Convolution sharding optimization tests
- `test_ttir_rmsnorm_sharding.py` - TTIR RMSNorm sharding tests
- `test_ttnn_rmsnorm_sharding.py` - TTNN RMSNorm sharding tests

### Subdirectory: `experimental/`

- `test_mpmd_ops.py` - MPMD (Multiple Program Multiple Data) operations

---

## 2. TTNN-JIT Tests (`test/ttnn-jit/`)

**Total: 21 test files**

These tests use TTNN's JIT (Just-In-Time) compilation system and execute directly on devices using the `device` fixture that calls `ttnn.open_device()`. They test the TTNN runtime's ability to trace, compile, and execute operations on hardware.

### Smoke Tests (Root Level)

Quick validation tests for core functionality:

- `test_eltwise_smoketest.py` - Element-wise operation smoke tests
- `test_eltwise_composite_smoketest.py` - Composite element-wise operations
- `test_matmul_smoketest.py` - Matrix multiplication smoke tests
- `test_reduction_smoketest.py` - Reduction operation smoke tests

### Other Root Level Tests

- `test_mesh_tensor_eltwise.py` - Multi-device mesh tensor operations
- `test_mixed_legacy_sharding_types.py` - Legacy sharding type compatibility
- `test_output_layouts.py` - Output tensor layout tests
- `test_program_cache.py` - Program caching functionality
- `test_unsupported_tensor_layouts.py` - Unsupported layout error handling

### Subdirectory: `nightly/`

Comprehensive nightly test suite:

- `test_eltwise.py` - Full element-wise operation test suite
- `test_eltwise_composite.py` - Composite element-wise operation suite
- `test_matmul.py` - Full matrix multiplication test suite
- `test_reduction.py` - Full reduction operation suite
- `test_layouts.py` - Tensor layout transformation suite

### Subdirectory: `lit/`

LLVM-style lit-driven tests:

- `test_ops.py` - General operation tests
- `test_control_flow.py` - Control flow operation tests
- `test_math_fidelity.py` - Mathematical precision tests
- `test_tracing_ir.py` - IR tracing functionality tests

### Subdirectory: `demo/`

Demonstration tests:

- `test_cosh.py` - Hyperbolic cosine operation demo
- `test_digamma.py` - Digamma function demo
- `test_multiply_accumulate.py` - Multiply-accumulate operation demo

---

## 3. PyKernel Demo Tests (`test/pykernel/demo/`)

**Total: ~17 files**

These are demonstration tests for Python-based kernels running on Tenstorrent devices. They showcase low-level kernel development capabilities.

**Example files**:
- `eltwise_sfpu_demo.py` - Element-wise SFPU (Special Function Processing Unit) operations
- `matmul_singlecore_demo.py` - Single-core matrix multiplication
- `matmul_multicore_demo.py` - Multi-core matrix multiplication
- `vecadd_multicore_demo.py` - Multi-core vector addition
- `dprint_demo.py` - Device print functionality demo

---

## How to Run Device Tests

### Prerequisites

```bash
# Navigate to tt-mlir directory
cd /localdev/ndrakulic/tt-xla/third_party/tt-mlir/src/tt-mlir

# Activate the environment
source env/activate

# Generate system descriptor (REQUIRED for golden tests)
ttrt query --save-artifacts

# Set system descriptor path
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
```

### Running Golden Tests

```bash
# Run all golden tests
pytest test/python/golden/

# Run specific test file
pytest test/python/golden/test_metal_matmul.py

# Run specific category
pytest test/python/golden/ttir_ops/eltwise/
pytest test/python/golden/ttnn_ops/

# Run with verbose output
pytest -svv test/python/golden/test_metal_matmul.py

# Run optimizer tests (requires special flag)
pytest --require-opmodel test/python/golden/optimizer/

# Run with specific backend
pytest test/python/golden/test_ttir_ops.py -k "ttmetal"

# Run multi-chip tests (N300)
pytest test/python/golden/test_shardy_ops_n300.py
```

### Running TTNN-JIT Tests

```bash
# Run all TTNN-JIT tests
pytest test/ttnn-jit/

# Run smoke tests only
pytest test/ttnn-jit/test_eltwise_smoketest.py
pytest test/ttnn-jit/test_matmul_smoketest.py

# Run nightly tests
pytest test/ttnn-jit/nightly/

# Run specific test with verbose output
pytest -svv test/ttnn-jit/test_mesh_tensor_eltwise.py
```

### Running PyKernel Demo Tests

```bash
# Run pykernel demos
pytest test/pykernel/demo/
```

### Useful Test Options

```bash
# Save artifacts (flatbuffers, MLIR files)
pytest --save-artifacts test/python/golden/

# Print IR during compilation
pytest --print-ir test/python/golden/test_metal_matmul.py

# Skip execution (compile only, don't run on device)
pytest --skip-exec test/python/golden/

# Disable golden comparison (use random inputs)
pytest --disable-golden test/python/golden/

# Enable intermediate verification
pytest --enable-intermediate-verification test/python/golden/

# Dump memory after execution
pytest --dump-memory test/python/golden/

# Check absolute/relative tolerance
pytest --check-atol --check-rtol test/python/golden/
```

---

## Test Markers and Configuration

### Platform/Target Markers

Tests can be marked to skip or run only on specific configurations:

- `@pytest.mark.skip_config(["ttnn", "n150"])` - Skip on specific platform/backend combo
- `@pytest.mark.only_config(["ttmetal", "n300"])` - Run only on specific config
- `@pytest.mark.skip_exec(["silicon"])` - Skip execution on silicon (compile only)

### Supported Platforms

- **Backends**: `ttnn`, `ttmetal`, `emitc`, `emitpy`
- **Systems**: `n150` (1 chip Wormhole), `n300` (2 chip Wormhole), `p150` (1 chip Blackhole), `p300` (2 chip Blackhole), `llmbox` (8 chip), `tg`
- **Environments**: `silicon` (actual hardware), `sim` (simulator)

### Frontend Markers

- `@pytest.mark.frontend("ttir")` - Test starts from TTIR
- `@pytest.mark.frontend("shlo")` - Test starts from StableHLO

---

## Key Differences: Compiler-Only vs Device Tests

### Compiler-Only Tests (NOT included in this report)

Located in `test/ttmlir/` directory:
- `*.mlir` files with lit tests
- Use `llvm-lit` to run
- Only test compilation pipeline, no device execution
- Example: `test/ttmlir/Silicon/TTNN/simple_multiply.mlir`

### Device Tests (included in this report)

- Execute compiled code on actual Tenstorrent hardware
- Use `pytest` framework
- Require physical device or simulator
- Generate and execute flatbuffer binaries
- Compare results against golden outputs
- Test both compilation AND runtime behavior

---

## Test Fixtures

### Golden Tests (`test/python/golden/conftest.py`)

- **`device` fixture**: Opens mesh device via `tt_runtime.runtime.open_mesh_device()`
- Supports multiple backends: `ttnn`, `ttmetal`, `emitc`, `emitpy`
- Supports mesh shapes for multi-chip testing
- Manages device lifecycle (open/close)
- Configures fabric for multi-chip communication

### TTNN-JIT Tests (`test/ttnn-jit/conftest.py`)

- **`device` fixture**: Opens single device via `ttnn.open_device()`
- **`mesh_device` fixture**: Opens multi-device mesh via `ttnn.open_mesh_device()`
- Disables program cache for clean test runs
- Handles dispatch core configuration (ETH vs WORKER)

---

## Environment Variables

### Required
- `SYSTEM_DESC_PATH` - Path to system descriptor file (golden tests)
- `TT_MLIR_HOME` - TT-MLIR installation directory
- `TT_METAL_RUNTIME_ROOT` - TT-Metal runtime directory

### Optional
- `TT_METAL_SIMULATOR` - Set to enable simulator mode instead of silicon
- `TTMLIR_ENABLE_PERF_TRACE` - Enable performance tracing

---

## Report Generation Date

Generated: 2026-02-15

## Additional Notes

1. All device tests require either physical Tenstorrent hardware or the simulator to be available
2. Golden tests compile through the full MLIR → Flatbuffer → Device pipeline
3. TTNN-JIT tests use TTNN's tracing and JIT compilation for faster iteration
4. Test execution order is sorted by target backend for efficient device reuse
5. Device handles are cached and reused across tests when possible to reduce overhead
6. Multi-chip tests automatically skip if insufficient devices are available
