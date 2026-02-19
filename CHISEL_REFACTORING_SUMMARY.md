# Chisel Tool Refactoring Summary: TTNN-Based Goldens

## Overview

Successfully refactored the chisel differential debugging tool from a dual-IR architecture (TTIR golden vs TTNN device) to a single-IR architecture (TTNN for both golden/CPU and device/hardware execution).

## Key Changes

### Architecture Transformation

**Before:**
```
Input: TTIR file
   ↓
chisel_pipeline: TTIR → TTNN compilation
   ↓
┌─────────────────┬─────────────────┐
│ Golden Path     │ Device Path     │
│ (TTIR on CPU)   │ (TTNN on HW)    │
├─────────────────┼─────────────────┤
│ GoldenExecutor  │ TTRT Runtime    │
│ + TTIR goldens  │ + Flatbuffer    │
└─────────────────┴─────────────────┘
```

**After:**
```
Input: TTNN flatbuffer
   ↓
Extract MLIR from flatbuffer (lazy on first preop)
   ↓
┌─────────────────┬─────────────────┐
│ Golden Path     │ Device Path     │
│ (TTNN on CPU)   │ (TTNN on HW)    │
├─────────────────┼─────────────────┤
│ GoldenExecutor  │ TTRT Runtime    │
│ + TTNN goldens  │ + Flatbuffer    │
└─────────────────┴─────────────────┘
```

## Implementation Details

### Phase 1: Lazy Context Initialization with Global Variable Pattern

**File:** `runtime/tools/chisel/chisel/core/context.py`

**Changes:**
1. Added global `_chisel_context` variable for lazy initialization
2. Refactored `ChiselContext.__init__()` to be lightweight (no MLIR loading)
3. Added `initialize_from_flatbuffer()` method that:
   - Extracts MLIR using `ttrt.binary.mlir_as_dict(binary)`
   - Parses MLIR to Module
   - Creates IRModule wrappers for both GOLDEN and DEVICE
   - Initializes Registry and GoldenExecutor
   - Loads all operations

4. Converted `preop()` and `postop()` from instance methods to module-level functions
5. Added `handle_preop()` and `handle_postop()` instance methods for actual work
6. Added helper functions:
   - `setup_chisel(**kwargs)`: Create global context
   - `bind_chisel_callbacks()`: Register callbacks with `DebugHooks.get()`

**Benefits:**
- ✅ Single source of truth (MLIR from flatbuffer)
- ✅ Lazy initialization (faster startup)
- ✅ Cleaner callback API (module-level functions)
- ✅ No file synchronization issues

### Phase 2: Refactor GoldenExecutor for TTNN

**File:** `runtime/tools/chisel/chisel/core/golden_executor.py`

**Changes:**
1. Updated all docstrings to reflect TTNN (not TTIR) operations
2. Removed TTIR-specific special case handling:
   - `ttir.empty` special case (line 112-113)
   - `ttir.dot_general` special handling (lines 156-200)
   - `ttir.broadcast` special handling (lines 201-214)
   - `ttir.pad` special handling (lines 215-229)
   - `ttir.permute` special handling (lines 230-236)

3. Simplified `execute()` method to use standardized golden function lookup
4. All TTNN operations now use: `golden_fn(*inputs) if inputs else golden_fn()`

**Benefits:**
- ✅ Simpler code (removed ~80 lines of special-case handling)
- ✅ Consistent operation execution
- ✅ Better maintainability

### Phase 3: Update Registry for Single TTNN Module

**File:** `runtime/tools/chisel/chisel/core/registry.py`

**Changes:**
1. Updated module docstring to reflect TTNN-only operations
2. Updated class docstrings to clarify that both modules wrap same TTNN MLIR
3. Updated `_merge_empty_golden_groups()` documentation
4. Clarified that GOLDEN and DEVICE ExecutionTypes now both use TTNN

**Note:** Registry structure unchanged - still accepts `golden_module` and `device_module` parameters, but both now wrap the same underlying TTNN MLIR module.

**Benefits:**
- ✅ Clearer documentation
- ✅ Same operations in both contexts
- ✅ Simpler debugging (same IR dialect)

### Phase 4: Simplify CLI and Main Workflow

**File:** `runtime/tools/chisel/chisel/main.py`

**Changes:**
1. **Removed arguments:**
   - `--input-file` (no longer needed)
   - `--dump-ttir` (no TTIR compilation)
   - `--dump-ttnn` (can dump from flatbuffer directly)

2. **Updated arguments:**
   - `--flatbuffer-path` now required with `-f` short option
   - Updated description to "TTNN differential debugging tool"

3. **Simplified main() function:**
   - Removed `chisel_pipeline()` usage
   - Removed `ttnn_to_flatbuffer_file()` call
   - Use `setup_chisel()` and `bind_chisel_callbacks()`
   - Use `ttrt.runtime.submit()` for execution
   - MLIR extracted automatically on first preop

4. **Removed imports:**
   - `chisel_pipeline`
   - `ChiselContext` (imported from context module instead)
   - `ExecutionType`
   - `ttnn_to_flatbuffer_file`

**Benefits:**
- ✅ Simpler user experience (only flatbuffer needed)
- ✅ Clearer purpose (differential debugging, not compilation)
- ✅ Fewer dependencies

### Phase 5: ExecutionType Enum

**Status:** Kept unchanged for backwards compatibility

**Decision:** Retain `ExecutionType.GOLDEN` and `ExecutionType.DEVICE` naming. Both now work with TTNN operations, differing only in execution backend (CPU vs hardware).

## Usage

### Before (Old Workflow)
```bash
# Required TTIR input file
chisel --input-file test.mlir --flatbuffer-path test.ttnn
```

### After (New Workflow)
```bash
# Only flatbuffer required (MLIR extracted automatically)
chisel -f test.ttnn --output-dir results/ --report-path results/report.csv
```

## Testing

Created test files:
- `test/test_ttnn_add.mlir`: Simple TTNN test case
- `test/test_chisel_ttnn.py`: End-to-end test script

**Test workflow:**
1. Compile TTNN MLIR → flatbuffer with embedded MLIR
2. Run chisel with flatbuffer-only input
3. Verify report generation and output artifacts

## Benefits of Refactoring

1. **Single Source of Truth**
   - MLIR extracted from flatbuffer
   - No MLIR/flatbuffer synchronization issues

2. **Simpler Architecture**
   - One IR dialect (TTNN) instead of two (TTIR + TTNN)
   - Same operations in both golden and device paths
   - Removed ~100 lines of TTIR-specific code

3. **More Accurate Comparisons**
   - Compare same TTNN operations on different backends
   - Isolates hardware issues from compilation issues

4. **Better User Experience**
   - Single input file (flatbuffer)
   - Clearer tool purpose
   - Faster startup (lazy initialization)

5. **Easier Maintenance**
   - Less special-case code
   - Better separation of concerns
   - Clearer documentation

## Files Modified

### Core Changes
1. `runtime/tools/chisel/chisel/core/context.py` (+150 lines, refactored)
2. `runtime/tools/chisel/chisel/core/golden_executor.py` (-80 lines, simplified)
3. `runtime/tools/chisel/chisel/core/registry.py` (documentation updates)
4. `runtime/tools/chisel/chisel/main.py` (-40 lines, simplified)

### Test Files Created
5. `runtime/tools/chisel/test/test_ttnn_add.mlir` (new)
6. `runtime/tools/chisel/test/test_chisel_ttnn.py` (new)

## Backwards Compatibility

**Breaking changes:**
- `--input-file` argument removed (use flatbuffer-only)
- `--dump-ttir` and `--dump-ttnn` removed
- `ChiselContext` constructor signature changed

**Migration guide:**
```python
# OLD
chisel_context = ChiselContext(
    ttir_module=ttir_module,
    ttnn_module=ttnn_module,
    output_dir=output_dir,
    ...
)
chisel_context.bind_callbacks()
chisel_context.run()

# NEW
from chisel.core.context import setup_chisel, bind_chisel_callbacks
from ttrt.runtime import submit
from ttrt.binary import load_binary_from_path

setup_chisel(
    output_dir=output_dir,
    report_path=report_path,
    ...
)
bind_chisel_callbacks()

binary = load_binary_from_path(flatbuffer_path)
submit(binary, program_index, [], [])
```

## Important: Embedding MLIR in Flatbuffers

**CRITICAL**: The flatbuffer must contain embedded MLIR source for chisel to work!

### Using Python API (Recommended)

```python
from ttmlir.ir import Context, Module
from ttmlir.passes import ttnn_to_flatbuffer_file

# Read and parse MLIR
ctx = Context()
ctx.load_all_available_dialects()
with open("input.mlir", "r") as f:
    mlir_source = f.read()
module = Module.parse(mlir_source, ctx)

# Create flatbuffer with embedded MLIR
# The 4th parameter (moduleCache) embeds the MLIR source
module_cache = [("ttnn", mlir_source)]  # List of (name, source) tuples

ttnn_to_flatbuffer_file(
    module,
    "output.ttnn",
    {},  # goldenMap (empty)
    module_cache  # Embeds MLIR in flatbuffer
)
```

### Verification

```python
from ttrt.binary import load_binary_from_path, mlir_as_dict

binary = load_binary_from_path("output.ttnn")
mlir_dict = mlir_as_dict(binary)

# Access MLIR using the name from module_cache ("ttnn")
if mlir_dict.get('ttnn'):
    print(f"✓ MLIR embedded: {len(mlir_dict['ttnn'])} bytes")
    print(f"  Module cache keys: {list(mlir_dict.keys())}")
else:
    print("✗ No MLIR found in flatbuffer!")
    print(f"  Available keys: {list(mlir_dict.keys())}")
```

**Note**: The key used to access the MLIR must match the name in `module_cache`. Since we use `("ttnn", mlir_source)`, we access it with `mlir_dict['ttnn']`.

### Running the Test

```bash
# 1. Build the project
cmake --build build

# 2. Ensure system descriptor exists
ttrt query --save-artifacts

# 3. Run the automated test
cd runtime/tools/chisel/test
python test_chisel_ttnn.py
```

The test script automatically creates a flatbuffer with embedded MLIR and verifies the chisel implementation.

## Future Enhancements

1. **Optional MLIR file input**: Add `--mlir-file` fallback for flatbuffers without embedded MLIR
2. **Input tensor loading**: Support `--load-inputs-from-disk` in flatbuffer-only mode
3. **Class rename**: Optionally rename `GoldenExecutor` → `CPUExecutor` for clarity
4. **Enhanced reporting**: Add more detailed comparison metrics
5. **Performance tracing**: Integrate with TT-MLIR performance tracing

## Conclusion

Successfully refactored chisel from TTIR-based goldens to TTNN-based goldens with:
- ✅ Lazy initialization from flatbuffer MLIR
- ✅ Global variable pattern for callbacks
- ✅ Simplified golden executor
- ✅ Single-IR architecture
- ✅ Cleaner CLI and user experience

The tool is now simpler, more accurate, and easier to maintain while providing the same differential debugging capabilities.
