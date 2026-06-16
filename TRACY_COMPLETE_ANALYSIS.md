# Tracy Profiling Empty CSV - Complete Diagnosis

## Executive Summary

The `ops_perf_results_*.csv` file is empty because of a **three-layer incompatibility** between tt-mlir's profiling implementation and the Tracy post-processing tools from tt-metal.

## The Three-Layer Problem

### Layer 1: Zone Naming (tracy_ops_times.csv generation)
**Problem**: Tracy CSV export filters zones by prefix
- **Tracy export command**: `csvexport -u -p TT_` (filters for "TT_" prefix)
- **tt-mlir zones**: Use generic names like "EnqueueProgramCommand", "CreateBufferCommand"
- **Result**: tracy_ops_times.csv has 422,981 lines but filtered out tt-mlir zones

### Layer 2: Message Format (tracy_ops_data.csv generation)
**Problem**: tt-mlir doesn't emit operation tracking messages
- **Expected format**: Tracy messages with JSON operation data (TT_DNN/TT_METAL OP messages)
- **tt-mlir emits**: Only `MLIR_PROGRAM_METADATA` metadata message
- **Result**: tracy_ops_data.csv has only 1 message, no operation data

### Layer 3: Post-Processing Filter
**Problem**: Post-processor filters by message prefix
- **Post-processor looks for**: "TT_DNN" or "TT_METAL" in messages
- **tt-mlir messages**: "MLIR_PROGRAM_METADATA" (partially fixed with patch)
- **Result**: import_tracy_op_logs() returns 0 ops, empty CSV generated

## Why Device Profiling Works

Device-side profiling bypasses all these layers:
- Goes directly to device firmware instrumentation
- Uses different data collection path
- Works independently of host-side Tracy zones/messages
- **Result**: profile_log_device.csv has 298 lines of RISC processor data

## Root Cause

**tt-mlir's profiling implementation was not designed to integrate with tt-metal's Tracy post-processing pipeline.**

The tt-mlir runtime:
1. Uses Tracy zones for timing (good ✓)
2. BUT uses generic zone names without "TT_" prefix (incompatible ✗)
3. Emits metadata messages with "MLIR_" prefix (incompatible ✗)
4. Does NOT emit per-operation JSON tracking messages (missing ✗)

The tt-metal Tracy tools expect:
1. Zones prefixed with "TT_"
2. Tracy messages for operations containing "TT_DNN" or "TT_METAL"
3. JSON-encoded operation details in messages
4. Specific message formats for ops, traces, and signposts

## What Needs to be Fixed

### For Host-Side Operation Profiling to Work:

#### Option A: Modify tt-mlir Runtime (Comprehensive Fix)
1. **Add TT_ prefix to zones**:
   ```cpp
   // In runtime/lib/ttmetal/executor.cpp
   ZoneScopedN("TT_MLIR_EnqueueProgramCommand");  // Was: "EnqueueProgramCommand"
   ```

2. **Emit operation tracking messages**:
   ```cpp
   // When executing an operation, emit Tracy message with JSON data:
   std::string opJson = R"({
     "global_call_count": )" + std::to_string(opId) + R"(,
     "op_code": "mlir.matmul",
     "device_id": 0
   })";
   TracyMessage(("TT_MLIR_OP ->\n" + opJson).c_str(), ...);
   ```

3. **Follow tt-metal operation message format**:
   - Message name: `TT_MLIR_OP` (or `TT_MLIR_DNN_OP`)
   - Format: `"TT_MLIR_OP ->\n{json_data}"`
   - Required JSON fields: `global_call_count`, `op_code`, timestamps, etc.

#### Option B: Create Custom tt-mlir Post-Processor (Simpler)
1. Fork or extend Tracy's process_ops_logs.py
2. Add tt-mlir specific message parsing
3. Generate tt-mlir specific performance reports
4. Don't rely on tt-metal's format expectations

#### Option C: Hybrid Approach (Best for Now)
1. Use device profiling for kernel performance (already working)
2. Add custom Python-level profiling decorators for host operations
3. Generate custom reports from both sources
4. Incrementally add tt-metal compatibility

## Immediate Actions

### What You Can Do Right Now:

1. **Use Device Profiling**:
   ```bash
   # Device profiling works and provides valuable data
   head -50 prof/reports/2026_06_16_15_20_05/profile_log_device.csv
   ```

2. **Check Tracy Zones Captured**:
   ```bash
   # See what timing data was actually captured
   head -100 prof/.logs/tracy_ops_times.csv
   ```

3. **Process Device-Only Data**:
   ```bash
   python -m tracy.process_ops_logs --output-folder prof --device-only
   ```

### What the Team Should Do:

1. **Decide on Integration Strategy**:
   - Full tt-metal compatibility (Option A)?
   - Custom tt-mlir tooling (Option B)?
   - Hybrid approach (Option C)?

2. **Document Profiling Architecture**:
   - How should tt-mlir operations be tracked?
   - What performance metrics are important?
   - How to integrate with existing tools?

3. **Implement Solution**:
   - Add proper Tracy instrumentation to runtime
   - Create or adapt post-processing tools
   - Add tests for profiling functionality

## Files Reference

**tt-mlir Runtime**:
- `runtime/include/tt/runtime/perf.h` - Profiling interface
- `runtime/lib/common/perf.cpp` - Tracy message emission
- `runtime/lib/ttmetal/executor.cpp` - Zone instrumentation
- `runtime/lib/ttnn/program_executor.cpp` - More zones

**Tracy Tools** (in `/opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/tracy/`):
- `__main__.py` - Main tracy profiler entry point
- `__init__.py` - Report generation (line ~140 for CSV export commands)
- `process_ops_logs.py` - Post-processing (line ~365 for message filtering)

**Generated Data**:
- `prof/.logs/tracy_profile_log_host.tracy` - Binary Tracy capture
- `prof/.logs/tracy_ops_times.csv` - 422,981 lines (filtered, no TT_ zones)
- `prof/.logs/tracy_ops_data.csv` - 1 message (MLIR_PROGRAM_METADATA only)
- `prof/reports/.../ops_perf_results_*.csv` - Empty (no ops to process)
- `prof/reports/.../profile_log_device.csv` - 298 lines (device profiling works!)

## Next Steps

1. Review this analysis with the team
2. Choose an integration approach
3. File issues/tasks for implementation
4. Consider creating a tt-mlir profiling design doc
