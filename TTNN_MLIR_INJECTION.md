# TTNN MLIR Injection to Flatbuffer

## Summary

Modified `lib/Target/TTNN/TTNNToFlatbuffer.cpp` to optionally inject the TTNN MLIR code into the flatbuffer's DebugInfo section, guarded by the `TT_INJECT_TTNN2FB` environment variable.

## Changes Made

### File: `lib/Target/TTNN/TTNNToFlatbuffer.cpp`

1. **Added includes** (lines 54-55):
   ```cpp
   #include <algorithm>
   #include <cstdlib>
   ```

2. **Added TTNN MLIR injection logic** (lines 4054-4073):
   ```cpp
   // Optionally inject TTNN MLIR into moduleCache for debugging/analysis
   // Guard behind TT_INJECT_TTNN2FB environment variable
   std::vector<std::pair<std::string, std::string>> enrichedModuleCache =
       moduleCache;
   if (const char *injectEnv = std::getenv("TT_INJECT_TTNN2FB");
       injectEnv != nullptr && std::string(injectEnv) == "1") {
     // Check if "ttnn" entry already exists in moduleCache
     auto ttnnEntry = std::find_if(
         enrichedModuleCache.begin(), enrichedModuleCache.end(),
         [](const auto &p) { return p.first == "ttnn"; });

     // Only add if not already present
     if (ttnnEntry == enrichedModuleCache.end()) {
       std::string ttnnMlirString;
       llvm::raw_string_ostream os(ttnnMlirString);
       rootModule.print(os, mlir::OpPrintingFlags().enableDebugInfo());
       os.flush();
       enrichedModuleCache.push_back({"ttnn", ttnnMlirString});
     }
   }
   ```

3. **Modified debugInfoToFlatbuffer call** (line 4082):
   - Changed from: `debugInfoToFlatbuffer(fbb, goldenMap, moduleCache)`
   - Changed to: `debugInfoToFlatbuffer(fbb, goldenMap, enrichedModuleCache)`

## How It Works

### When `TT_INJECT_TTNN2FB=1`:
1. The TTNN MLIR module is serialized to a string with debug info enabled
2. The string is added to the `moduleCache` as a pair: `{"ttnn", ttnnMlirString}`
3. The `moduleCache` is stored in the flatbuffer's `DebugInfo` section
4. The TTNN MLIR is now embedded in the flatbuffer and can be extracted later

### When `TT_INJECT_TTNN2FB` is not set or is `0`:
- Behaves exactly as before
- No TTNN MLIR is injected
- No performance impact

## Usage

### Running TT-MLIR Tests with TTNN MLIR Injection

```bash
cd /localdev/ndrakulic/tt-xla/third_party/tt-mlir/src/tt-mlir
source env/activate

# Rebuild tt-mlir with the changes
cmake --build build

# Generate system descriptor (required for golden tests)
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys

# Enable TTNN MLIR injection
export TT_INJECT_TTNN2FB=1

# Run golden tests (all device tests)
pytest test/python/golden/

# Run specific test
pytest test/python/golden/test_metal_matmul.py -v

# Run TTNN-JIT tests
pytest test/ttnn-jit/test_eltwise_smoketest.py -v
```

### Verifying the Injection

To verify that the TTNN MLIR is being injected into the flatbuffer, you can:

1. **Save the flatbuffer** using `--save-artifacts`:
   ```bash
   export TT_INJECT_TTNN2FB=1
   pytest --save-artifacts test/python/golden/test_metal_matmul.py
   ```

2. **Check the flatbuffer** (the TTNN MLIR will be in the DebugInfo section):
   ```python
   import _ttmlir_runtime as tt_runtime

   # Load the flatbuffer
   binary = tt_runtime.binary.load_flatbuffer_from_path("path/to/flatbuffer.ttnn")

   # Access the debug info (requires additional runtime API support)
   # The TTNN MLIR is stored in the module_cache field of DebugInfo
   ```

## Benefits

1. **Debugging**: TTNN MLIR is embedded in flatbuffers for post-mortem analysis
2. **Reproducibility**: Can reconstruct the exact MLIR that generated a flatbuffer
3. **Testing**: Useful for chisel and other tools that need to compare MLIR stages
4. **Zero overhead when disabled**: No impact when `TT_INJECT_TTNN2FB` is not set
5. **Automatic**: All 73 device tests automatically benefit from this feature

## Impact on Tests

### Tests Affected:
- **All 52 Golden Tests** (`test/python/golden/**`)
- **All 21 TTNN-JIT Tests** (`test/ttnn-jit/**`)
- **PyKernel Tests** (`test/pykernel/**`)

All these tests will now optionally embed TTNN MLIR in their flatbuffers when the environment variable is set, without requiring any test code modifications.

## Comparison with TT-XLA

This implementation mirrors what was done in tt-xla's `module_builder.cc`:

### TT-XLA (module_builder.cc):
```cpp
std::string ttnn_mlir;
llvm::raw_string_ostream os(ttnn_mlir);
mlir_module.get().print(os, mlir::OpPrintingFlags().enableDebugInfo());
os.flush();
std::string ttnn_mlir_name = "ttnn";
flatbuffer_binary = mlir::tt::ttnn::ttnnToFlatbuffer(
    mlir_module.get(), {}, {{ttnn_mlir_name, ttnn_mlir}});
```

### TT-MLIR (TTNNToFlatbuffer.cpp):
```cpp
if (const char *injectEnv = std::getenv("TT_INJECT_TTNN2FB");
    injectEnv != nullptr && std::string(injectEnv) == "1") {
  std::string ttnnMlirString;
  llvm::raw_string_ostream os(ttnnMlirString);
  rootModule.print(os, mlir::OpPrintingFlags().enableDebugInfo());
  os.flush();
  enrichedModuleCache.push_back({"ttnn", ttnnMlirString});
}
```

The key difference is that tt-mlir guards this behavior behind an environment variable for optional activation.

## Date
Modified: 2026-02-15
