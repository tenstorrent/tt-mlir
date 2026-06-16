# Performance Data Available in Prof Directory - Summary

## Quick Answer

**YES** - There is enough information to calculate performance metrics!

## What You Have

### Primary Source: `cpp_device_perf_report.csv`
```
prof/.logs/cpp_device_perf_report.csv
```

This file contains everything needed for device performance analysis:
- ✅ Kernel execution time (894.207 ms for your matmul test)
- ✅ Per-RISC processor breakdown
- ✅ Core utilization (4 cores used out of 64)
- ✅ Firmware overhead
- ✅ Device architecture info

### Analysis Tool Created
```bash
source env/activate
python tools/scripts/analyze_prof_data.py
```

### Output Format
```json
{
  "summary": {
    "total_operations": 1,
    "total_kernel_duration_ms": 894.207,
    "total_kernel_duration_s": 0.894,
    "avg_kernel_duration_ms": 894.207
  },
  "operations": [
    {
      "kernel_duration_ms": 894.207,
      "cores_used": 4,
      "available_cores": 64,
      "risc_breakdown": {
        "trisc0": { "duration_ms": 894.207 },
        "trisc1": { "duration_ms": 894.206 },
        "trisc2": { "duration_ms": 894.207 },
        "ncrisc": { "duration_ms": 894.206 },
        "brisc": { "duration_ms": 0.0003 }
      }
    }
  ]
}
```

## Key Metrics Calculated

1. **Total Kernel Duration**: 894.207 ms (0.894 seconds)
2. **Core Utilization**: 6.25% (4 of 64 cores)
3. **Compute Utilization**: 100% (all TRISCs maxed out)
4. **Memory Utilization**: 100% (NCRISC maxed out)
5. **Data Movement**: Minimal (BRISC ~0 ms)

## What's Missing vs tt-metal Standard Reports

The standard tt-metal profiling expects:
- Host-side operation names (e.g., "ttnn.matmul")
- Operation metadata (shapes, dtypes, etc.)
- TT_DNN/TT_METAL message tracking

What you have instead:
- Device kernel performance (same underlying perf data)
- Direct-to-metal execution metrics
- No TTNN API layer (expected for d2m-jit)

## Files in Prof Directory

```
prof/
├── .logs/
│   ├── cpp_device_perf_report.csv       ✅ PRIMARY - All device perf metrics
│   ├── profile_log_device.csv           ✅ Detailed per-core timing (298 lines)
│   ├── tracy_ops_times.csv              ✅ Host-side timing (424K lines)
│   ├── tracy_ops_data.csv               ⚠️  Only metadata (MLIR_PROGRAM_METADATA)
│   └── tracy_profile_log_host.tracy     ✅ Binary Tracy capture
├── reports/
│   └── 2026_06_16_15_49_36/
│       ├── ops_perf_results_*.csv       ❌ Empty (expected for d2m-jit)
│       ├── per_core_op_to_op_times.csv  ⚠️  Minimal data
│       └── profile_log_device.csv       ✅ Device profiling (copy)
└── perf_analysis.json                    ✅ Generated analysis
```

## How tt-perf-report Works (for reference)

Standard tt-metal performance reports:
1. Read `ops_perf_results.csv` for operation list
2. Extract `DEVICE KERNEL DURATION [ns]` column
3. Sum durations for total time
4. Aggregate by operation type

Your data flow:
1. Read `cpp_device_perf_report.csv` directly ✅
2. Extract `DEVICE KERNEL DURATION [ns]` column ✅
3. Sum durations for total time ✅
4. No operation types (d2m compiles to raw kernels)

## Performance Insights from Your Test

**test_matmul_compiles_and_runs**:

- **Performance**: 894ms total execution
- **Bottleneck**: Compute-bound (all TRISCs at 100%)
- **Memory**: Fully utilized (NCRISC at 100%)
- **Optimization opportunity**:
  - Only using 4 cores (6.25% of available 64)
  - Could potentially parallelize 16x
  - Limited by kernel launch configuration

## Next Steps

### For Performance Optimization:
1. Increase core utilization (launch on more cores)
2. Current: 4 cores → Target: 32-64 cores
3. Expected speedup: ~8-16x

### For Better Profiling:
1. Add D2M operation metadata to Tracy messages
2. Emit operation names during kernel execution
3. Would populate `ops_perf_results.csv` with D2M-specific data

### For Current Analysis:
1. Use `analyze_prof_data.py` script
2. Focus on `cpp_device_perf_report.csv` metrics
3. Track kernel duration as primary KPI
4. Monitor RISC breakdown for bottlenecks

## Documentation Created

1. **PERF_CALCULATION_ANALYSIS.md** - Complete performance analysis guide
2. **TRACY_COMPLETE_ANALYSIS.md** - Full profiling system diagnosis
3. **TRACY_PROFILING_DEBUG.md** - Debugging why CSV was empty
4. **tools/scripts/analyze_prof_data.py** - Automated analysis script
5. **prof/perf_analysis.json** - Parsed performance data

## Conclusion

You have **all the device performance data** needed to calculate and analyze performance. The `cpp_device_perf_report.csv` file is the authoritative source and contains:

- Kernel execution times ✅
- Core utilization ✅
- RISC processor breakdown ✅
- Performance metrics ✅

The only missing piece is high-level operation names, which is expected for d2m-jit workloads that compile directly to Metal kernels without going through the TTNN API layer.

**Bottom line**: Use `cpp_device_perf_report.csv` for all your performance calculations!
