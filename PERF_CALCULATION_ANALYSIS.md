# Performance Calculation from Prof Directory

## Summary

**YES, there is enough information in the prof directory to calculate performance metrics!**

The `cpp_device_perf_report.csv` file contains comprehensive device-side performance data that can be used to calculate key performance metrics even though the host-side operation tracking (`ops_perf_results.csv`) is empty.

## Available Data Sources

### 1. cpp_device_perf_report.csv (PRIMARY SOURCE) ✅
**Location**: `prof/.logs/cpp_device_perf_report.csv`

This file contains aggregated per-operation device performance metrics:

| Metric | Description | Available |
|--------|-------------|-----------|
| **DEVICE KERNEL DURATION [ns]** | Total kernel execution time | ✅ Yes |
| **DEVICE FW DURATION [ns]** | Firmware execution time | ✅ Yes |
| **DEVICE KERNEL FIRST TO LAST START [ns]** | Parallel execution span | ✅ Yes |
| **Per-RISC Breakdown** | BRISC, NCRISC, TRISC0/1/2 durations | ✅ Yes |
| **Core Utilization** | Cores used vs available | ✅ Yes |
| **Device Info** | Architecture, device ID | ✅ Yes |
| **Op-to-Op Latency** | Time between operations | ✅ Yes |

**Example from your test run**:
- Operation 0 (matmul kernel):
  - Kernel Duration: **894.207 ms**
  - Firmware Duration: **894.208 ms**
  - Cores Used: **4 out of 64 available**
  - Device: **wormhole_b0**
  - RISC Breakdown:
    - BRISC: 0.000 ms (data movement)
    - NCRISC: 894.206 ms (NOC/memory)
    - TRISC0: 894.207 ms (compute)
    - TRISC1: 894.206 ms (compute)
    - TRISC2: 894.207 ms (compute)

### 2. profile_log_device.csv (DETAILED SOURCE) ✅
**Location**: `prof/.logs/profile_log_device.csv` (298 lines)

This file contains raw per-core, per-RISC zone timing data:
- Individual ZONE_START/ZONE_END events
- Cycle-accurate timestamps
- Per-core breakdown
- Firmware vs kernel execution phases

### 3. tracy_ops_times.csv (HOST TIMING) ✅
**Location**: `prof/.logs/tracy_ops_times.csv` (424,650 lines)

Contains host-side Python function timing:
- Function call durations
- Thread information
- Execution timeline

## What Can Be Calculated

### ✅ Device Performance Metrics
1. **Total Kernel Execution Time**: Sum of all kernel durations
2. **Per-Operation Timing**: Individual operation performance
3. **RISC Processor Utilization**: Which processors are active
4. **Core Utilization**: How many cores are being used
5. **Firmware Overhead**: Time spent in firmware vs kernels
6. **Op-to-Op Latency**: Gaps between operations
7. **Parallel Execution Analysis**: From FIRST_TO_LAST_START metric

### ✅ Analysis Capabilities
- **Performance bottleneck identification** (which RISC is slowest?)
- **Core utilization efficiency** (4 cores used out of 64 = 6.25%)
- **Memory vs compute balance** (NCRISC vs TRISC comparison)
- **Firmware overhead analysis** (FW vs kernel duration)

### ❌ What's Missing (due to empty ops_perf_results.csv)
- Host-side operation names and types
- High-level TTNN/Metal operation tracking
- Host-to-device correlation
- Python-level profiling integration

## How to Calculate Performance

### Method 1: Use the Provided Script

```bash
source env/activate
python tools/scripts/analyze_prof_data.py
```

This script:
1. Reads `cpp_device_perf_report.csv`
2. Calculates summary metrics
3. Provides per-operation breakdown
4. Saves detailed JSON report to `prof/perf_analysis.json`

### Method 2: Manual CSV Processing

```python
import csv

with open('prof/.logs/cpp_device_perf_report.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        kernel_dur_ns = int(row["DEVICE KERNEL DURATION [ns]"])
        kernel_dur_ms = kernel_dur_ns / 1_000_000
        print(f"Kernel Duration: {kernel_dur_ms:.3f} ms")

        # RISC breakdown
        for risc in ["BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"]:
            key = f"DEVICE {risc} KERNEL DURATION [ns]"
            if row[key]:
                dur_ms = int(row[key]) / 1_000_000
                print(f"  {risc}: {dur_ms:.3f} ms")
```

### Method 3: Use Existing tt-mlir Tools

Several tools in the codebase can process this data:

1. **collect_perf_results.py**:
   ```python
   from tools.scripts.collect_perf_results import sum_device_kernel_duration
   total_ns = sum_device_kernel_duration(csv_path)
   ```

2. **Tracy process_ops_logs.py**:
   Can parse `cpp_device_perf_report.csv` if host ops were available

3. **perf_ci/conftest.py**:
   Example of aggregating `DEVICE KERNEL DURATION [ns]`

## Example Performance Report

From your test run (`test_matmul_compiles_and_runs`):

```
Performance Summary:
  Total Operations: 1
  Total Kernel Duration: 894.207 ms (0.894 seconds)
  Total FW Duration: 894.208 ms
  Average Kernel Duration: 894.207 ms

Operation 0 (Matmul):
  Cores Used: 4 / 64 (6.25% utilization)
  Device: wormhole_b0

  RISC Breakdown:
    BRISC (Data Movement): 0.000 ms (0.00%)
    NCRISC (NOC/Memory): 894.206 ms (99.99%)
    TRISC0 (Compute): 894.207 ms (100.00%)
    TRISC1 (Compute): 894.206 ms (99.99%)
    TRISC2 (Compute): 894.207 ms (100.00%)
```

**Key Insights**:
- All three TRISC processors active (compute bound)
- NCRISC fully utilized (memory transfers)
- BRISC minimal usage (setup only)
- Low core utilization (4/64 cores = opportunity for parallelization)

## Comparison with tt-perf-report

The standard tt-metal performance reporting tools look for:
1. Host-side operation tracking (TT_DNN/TT_METAL messages)
2. Operation names and metadata
3. Host-to-device correlation

What we have instead:
1. Device-side kernel performance (same underlying data)
2. Cycle-accurate RISC processor timing
3. Core utilization metrics

**Bottom line**: We have the **device execution performance data**, we just don't have the **host operation metadata** that would tell us which high-level TTNN operations these kernels correspond to.

For d2m-jit workloads, this is actually the right level of detail since:
- D2M compiles directly to Metal kernels
- No TTNN operation layer
- Focus is on kernel performance, not operation classification

## Recommendations

### For Your Use Case (d2m-jit profiling):

1. **Use cpp_device_perf_report.csv** as the primary perf metric source
2. **Focus on kernel duration** as the main KPI
3. **Analyze RISC breakdown** to identify bottlenecks (compute vs memory)
4. **Track core utilization** for scaling opportunities
5. **Use the provided analyze_prof_data.py script** for automated analysis

### If You Need Operation-Level Tracking:

Consider instrumenting the d2m-jit runtime to emit operation metadata:
```python
# In d2m kernel execution code:
import tracy
tracy.TracyMessage(f"D2M_OP: matmul, cores=4, op_id={op_id}")
```

Then modify the tracy post-processor to recognize `D2M_OP` messages.

## Files Created

1. **tools/scripts/analyze_prof_data.py**: Performance analysis script
2. **prof/perf_analysis.json**: Detailed JSON report with all metrics
3. **TRACY_COMPLETE_ANALYSIS.md**: Full diagnosis of profiling system
4. **TRACY_PROFILING_DEBUG.md**: Quick debugging guide

## Conclusion

**You have sufficient data to calculate performance metrics!**

The `cpp_device_perf_report.csv` contains all the device-side performance information needed for performance analysis. The main limitation is the lack of host-side operation names/metadata, which is expected for d2m-jit workloads that bypass the TTNN API layer.

For your specific test (`test_matmul_compiles_and_runs`), you can see that:
- The matmul kernel took **~894ms** to execute
- It used **4 cores** (6.25% utilization)
- All compute units (TRISC0/1/2) were fully utilized
- Memory bandwidth (NCRISC) was maxed out
- There's opportunity for **16x parallelization** (64 cores available)
