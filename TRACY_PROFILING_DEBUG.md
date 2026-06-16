# Tracy Profiling Debug Report

## Issue
The `ops_perf_results_2026_06_16_15_20_05.csv` file is empty after running:
```bash
python -m tracy -r -v --output-folder prof -m pytest test/d2m-jit/test_matmul.py::test_matmul_compiles_and_runs
```

## Root Cause Analysis

### What Worked
1. ✅ Tracy capture process started successfully
2. ✅ Tracy binary files were created (`tracy_profile_log_host.tracy`)
3. ✅ Tracy CSV exports were generated:
   - `tracy_ops_times.csv` - 9+ KB with Python function timing data
   - `tracy_ops_data.csv` - Contains only `MLIR_PROGRAM_METADATA` message

### What Failed
The `tracy_ops_data.csv` file contains:
```csv
MessageName;total_ns
MLIR_PROGRAM_METADATA;;12073809561
```

**The problem**: No `TT_DNN` or `TT_METAL` operations were captured!

### Why ops_perf_results.csv is Empty

The Tracy post-processing function `import_tracy_op_logs()` filters for operations with these patterns:
- `"TT_DNN"` in message name
- `"TT_METAL"` in message name
- `"TT_SIGNPOST"` in message name

Since none of these were found in the captured data:
```python
Number of ops: 0
Number of signposts: 0
Number of trace replays: 0
```

The CSV generation has no data to write, resulting in an empty file.

## Root Cause Identified! ✓

### Message Tag Mismatch

The tt-mlir runtime and Tracy post-processor use **incompatible message prefixes**:

**What tt-mlir runtime emits** (`runtime/include/tt/runtime/perf.h`):
```cpp
enum class TracyLogTag {
  MLIR_OP_LOCATION,
  MLIR_CONST_EVAL_OP,
  MLIR_PROGRAM_METADATA,
  MLIR_INPUT_LAYOUT_CONVERSION_OP
};
```

**What Tracy post-processor looks for** (`/opt/ttmlir-toolchain/.../tracy/process_ops_logs.py`):
```python
if "TT_DNN" in opDataStr or "TT_METAL" in opDataStr:
    # Process operation...
if "TT_SIGNPOST" in opDataStr:
    # Process signpost...
```

**Result**: The tracy_ops_data.csv contains `MLIR_PROGRAM_METADATA` but no `TT_DNN` or `TT_METAL` messages, so the post-processor filters out all operations and generates an empty CSV.

### Evidence
From the captured data:
```csv
MessageName;total_ns
MLIR_PROGRAM_METADATA;;12073809561
```

From the import function output:
```
Number of ops: 0
Number of signposts: 0
Number of trace replays: 0
```

## Solutions

### Option 1: Modify Tracy Post-Processor (Recommended)
Create a custom post-processor or patch the existing one to recognize MLIR_ prefixes:

**Location**: `/opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/tracy/process_ops_logs.py`

**Change** (around line 365):
```python
# Current code:
if "TT_DNN" in opDataStr or "TT_METAL" in opDataStr:

# Change to:
if "TT_DNN" in opDataStr or "TT_METAL" in opDataStr or "MLIR_" in opDataStr:
```

This would allow the post-processor to recognize tt-mlir runtime messages.

### Option 2: Update tt-mlir Runtime to Use TT_METAL Prefix
Modify the runtime to emit compatible message tags:

**Location**: `runtime/include/tt/runtime/perf.h` and `runtime/lib/common/perf.cpp`

**Change enum**:
```cpp
enum class TracyLogTag {
  TT_METAL_MLIR_OP_LOCATION,          // Was: MLIR_OP_LOCATION
  TT_METAL_MLIR_CONST_EVAL_OP,        // Was: MLIR_CONST_EVAL_OP
  TT_METAL_MLIR_PROGRAM_METADATA,     // Was: MLIR_PROGRAM_METADATA
  TT_METAL_MLIR_INPUT_LAYOUT_CONVERSION_OP  // Was: MLIR_INPUT_LAYOUT_CONVERSION_OP
};
```

This would make tt-mlir compatible with the existing Tracy tools.

### Option 3: Create Custom tt-mlir Post-Processor
Write a new post-processor specifically for tt-mlir profiling data that:
1. Recognizes MLIR_ prefixes
2. Generates tt-mlir specific performance reports
3. Integrates with the existing tooling

### Option 4: Use Device-Side Profiling Only (Current Workaround)
The device-side profiling is working and has valuable data:
```bash
# View device profiling data (298 lines of RISC processor events):
head -30 prof/reports/2026_06_16_15_20_05/profile_log_device.csv

# Process device-only data:
python -m tracy.process_ops_logs --output-folder prof --device-only
```

This provides kernel-level performance insights even without host-side op tracking.

## Quick Fix (Immediate Solution)

To make profiling work right now, patch the Tracy post-processor:

```bash
# Backup the original file
cp /opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/tracy/process_ops_logs.py /opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/tracy/process_ops_logs.py.bak

# Edit line ~365 to add MLIR_ prefix support
# Find:     if "TT_DNN" in opDataStr or "TT_METAL" in opDataStr:
# Replace:  if "TT_DNN" in opDataStr or "TT_METAL" in opDataStr or "MLIR_" in opDataStr:

# Re-run the post-processor on existing data
cd /home/jgrim/wh-01-src/tt-mlir
source env/activate
python -m tracy.process_ops_logs --output-folder prof
```

## Files Involved

1. **tt-mlir Runtime Profiling**:
   - `runtime/include/tt/runtime/perf.h` - TracyLogTag enum definition
   - `runtime/lib/common/perf.cpp` - Tracy message emission

2. **Tracy Post-Processor**:
   - `/opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/tracy/process_ops_logs.py` - Line ~365, import_tracy_op_logs() function
   - `/opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/tracy/__init__.py` - Report generation orchestration

3. **Generated Data**:
   - `prof/.logs/tracy_ops_data.csv` - Contains MLIR_PROGRAM_METADATA messages
   - `prof/.logs/tracy_ops_times.csv` - Has Python function timing (9+ KB)
   - `prof/reports/.../profile_log_device.csv` - Device RISC profiling (working, 298 lines)
