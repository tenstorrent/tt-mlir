# Performance Testing Infrastructure

This directory contains tools for batch performance testing and analysis of D2M-JIT operations.

## Tools

### 1. `d2m-jit-perf.sh` - Single Test Performance Profiler

Runs a single pytest test with Tracy profiling enabled and collects device performance metrics.

**Usage:**
```bash
tools/perf-analyzer/d2m-jit-perf.sh <test-to-run> <traits-to-instrument> <output-dir> [system-descriptor]
```

**Example:**
```bash
tools/perf-analyzer/d2m-jit-perf.sh \
    test/d2m-jit/test_pattern_eltwise.py::test_pattern_add_exp \
    device-zone \
    prof_single
```

**Profiler Traits:**
- `fpu` - Floating Point Unit operations
- `sfpu` - Special Function Processing Unit operations
- `init` - Initialization operations
- `unary` - Unary operations (single input)
- `binary` - Binary operations (two inputs)
- `ternary` - Ternary operations (three inputs)
- `device-zone` - Device zone operations (default)
- `trid-noc` - Tensix Thread ID and Network-on-Chip operations
- `layout` - Layout transformation operations
- `all` - All operations

---

### 2. `batch-perf-test.sh` - Batch Test Runner ⭐ NEW

Automatically discovers and runs all tests in a pytest file through the performance profiler, then generates a comprehensive comparison report.

**Usage:**
```bash
tools/perf-analyzer/batch-perf-test.sh <pytest-file> <output-base-dir> [traits] [system-descriptor]
```

**Example:**
```bash
# Run all tests in a file
tools/perf-analyzer/batch-perf-test.sh \
    test/d2m-jit/test_pattern_eltwise.py \
    prof_batch \
    device-zone

# View the report
cat prof_batch/performance_report.md
```

**What It Does:**
1. **Discovers Tests**: Uses pytest to find all tests in the specified file
2. **Runs Each Test**: Executes each test individually through `d2m-jit-perf.sh`
3. **Collects Metrics**: Parses `cpp_device_perf_report.csv` from each test
4. **Generates Reports**:
   - `performance_summary.csv` - Machine-readable CSV with all metrics
   - `performance_report.md` - Human-readable markdown report
   - `batch_run.log` - Detailed execution log

**Output Structure:**
```
prof_batch/
├── performance_summary.csv      # All test metrics in CSV format
├── performance_report.md        # Formatted report with rankings
├── batch_run.log                # Detailed execution log
├── test__test_add_exp/          # Individual test output
│   └── reports/
│       └── <timestamp>/
│           ├── cpp_device_perf_report.csv
│           ├── profile_log_device.csv
│           └── tracy_ops_times.csv
├── test__test_mul_div/
│   └── ...
└── ...
```

---

### 3. `analyze-perf-batch.py` - Advanced Analysis Tool ⭐ NEW

Provides in-depth statistical analysis of batch test results.

**Usage:**
```bash
# Analyze a single batch run
tools/perf-analyzer/analyze-perf-batch.py prof_batch/performance_summary.csv

# Compare multiple runs
tools/perf-analyzer/analyze-perf-batch.py \
    --compare \
    run1/performance_summary.csv \
    run2/performance_summary.csv

# Export to JSON
tools/perf-analyzer/analyze-perf-batch.py \
    prof_batch/performance_summary.csv \
    --export-json results.json

# Save analysis to file
tools/perf-analyzer/analyze-perf-batch.py \
    prof_batch/performance_summary.csv \
    --output analysis.md
```

**Features:**

#### Statistical Analysis
- Mean, median, standard deviation
- Percentiles (P25, P75, P90, P95, P99)
- Min/max values
- Outlier detection

#### Efficiency Metrics
- Time per kernel (ms/kernel)
- Most/least efficient tests
- Anomaly detection

#### Multi-Run Comparison
- Side-by-side performance comparison
- Delta calculations (% change)
- Regression detection

#### Data Export
- JSON format for programmatic access
- Markdown reports

---

## Workflow Examples

### Basic Performance Testing

```bash
# 1. Run batch tests on a test file
tools/perf-analyzer/batch-perf-test.sh \
    test/d2m-jit/test_pattern_eltwise.py \
    prof_eltwise \
    device-zone

# 2. View the basic report
cat prof_eltwise/performance_report.md

# 3. Get detailed statistics
tools/perf-analyzer/analyze-perf-batch.py \
    prof_eltwise/performance_summary.csv
```

### Comparing Before/After Changes

```bash
# 1. Run baseline tests
git checkout main
tools/perf-analyzer/batch-perf-test.sh \
    test/d2m-jit/test_pattern_eltwise.py \
    prof_baseline \
    device-zone

# 2. Run tests with your changes
git checkout feature-branch
tools/perf-analyzer/batch-perf-test.sh \
    test/d2m-jit/test_pattern_eltwise.py \
    prof_feature \
    device-zone

# 3. Compare results
tools/perf-analyzer/analyze-perf-batch.py \
    --compare \
    prof_baseline/performance_summary.csv \
    prof_feature/performance_summary.csv \
    --output comparison.md

# 4. Check for regressions
cat comparison.md | grep "+"  # Look for performance increases (slower)
```

### Testing Different Profiler Traits

```bash
# Run with different instrumentation levels
for trait in device-zone fpu sfpu all; do
    tools/perf-analyzer/batch-perf-test.sh \
        test/d2m-jit/test_pattern_eltwise.py \
        prof_${trait} \
        ${trait}
done

# Compare overhead of different instrumentation
tools/perf-analyzer/analyze-perf-batch.py \
    --compare \
    prof_device-zone/performance_summary.csv \
    prof_all/performance_summary.csv
```

### Continuous Performance Monitoring

```bash
#!/bin/bash
# save as: tools/perf-analyzer/nightly-perf.sh

DATE=$(date +%Y%m%d)
OUTPUT_DIR="prof_nightly_${DATE}"

# Run all D2M-JIT tests
tools/perf-analyzer/batch-perf-test.sh \
    test/d2m-jit/test_pattern_eltwise.py \
    ${OUTPUT_DIR} \
    device-zone

# Generate detailed analysis
tools/perf-analyzer/analyze-perf-batch.py \
    ${OUTPUT_DIR}/performance_summary.csv \
    --output ${OUTPUT_DIR}/analysis.md \
    --export-json ${OUTPUT_DIR}/results.json

# Compare with yesterday
YESTERDAY=$(date -d "yesterday" +%Y%m%d)
if [ -d "prof_nightly_${YESTERDAY}" ]; then
    tools/perf-analyzer/analyze-perf-batch.py \
        --compare \
        prof_nightly_${YESTERDAY}/performance_summary.csv \
        ${OUTPUT_DIR}/performance_summary.csv \
        --output ${OUTPUT_DIR}/vs_yesterday.md
fi

echo "Performance report: ${OUTPUT_DIR}/performance_report.md"
echo "Detailed analysis: ${OUTPUT_DIR}/analysis.md"
```

---

## Performance Metrics Explained

### Total Device Time (ms)
The sum of all kernel execution times for a test. Lower is better.

### Kernel Count
Number of kernels launched during the test. Fewer kernels often means better fusion.

### Average Kernel Time (ms)
Mean execution time per kernel. Useful for understanding kernel efficiency.

### Max/Min Kernel Time (ms)
The slowest and fastest kernels in the test. Large ranges may indicate optimization opportunities.

### Time per Kernel (ms/kernel)
Total time divided by kernel count. Good metric for overall efficiency.

---

## Interpreting Results

### Performance Report Sections

1. **Summary Statistics**: Overall metrics across all tests
2. **Detailed Results Table**: Per-test breakdown with status
3. **Performance Ranking**: Tests sorted by total device time
4. **Output Locations**: Where to find detailed data

### What to Look For

✅ **Good Performance Indicators:**
- Low total device time
- Few kernels (indicates good fusion)
- Consistent kernel times (low stddev)
- Time per kernel < 1ms

⚠️ **Performance Concerns:**
- High total device time compared to similar tests
- Many kernels (may indicate missed fusion opportunities)
- High variance in kernel times
- Outliers (tests much slower than P95)

### Common Analysis Tasks

**Find Slowest Tests:**
```bash
cat prof_batch/performance_report.md | grep -A 10 "Performance Ranking"
```

**Check for Failed Tests:**
```bash
grep FAILED prof_batch/performance_summary.csv
```

**Find Tests with Many Kernels:**
```bash
awk -F',' '$3 > 10' prof_batch/performance_summary.csv | column -t -s','
```

**Calculate Total Device Time:**
```bash
awk -F',' 'NR>1 {sum+=$2} END {print "Total:", sum, "ms"}' \
    prof_batch/performance_summary.csv
```

---

## Troubleshooting

### No Tests Collected
- Check that the pytest file path is correct
- Ensure tests are discoverable: `pytest <file> --collect-only`
- Verify pytest is installed in venv

### Missing Performance Data
- Check that `SYSTEM_DESC_PATH` is set correctly
- Verify device is available: `tt-smi`
- Look at individual test logs in `<output-dir>/<test>/`

### Tracy Profiler Errors
- Ensure `TT_METAL_DEVICE_PROFILER_DISPATCH=0` is set
- Check that tracy python module is installed
- See `batch_run.log` for detailed error messages

### Script Hangs
- Check for device timeouts
- Look for infinite loops in test code
- Kill hung processes: `pkill -f pytest`

---

## Tips and Best Practices

1. **Start Small**: Test with a few tests first before running full batches
2. **Use Consistent Traits**: Stick to `device-zone` for most comparisons
3. **Save Baselines**: Keep a baseline run to compare against future changes
4. **Clean Between Runs**: Use `rm -rf <output-dir>` to start fresh
5. **Check Logs**: Always review `batch_run.log` for errors
6. **Watch Resources**: Monitor disk space and memory during large batch runs

---

## Related Documentation

- [Tracy Profiling Analysis](../../TRACY_COMPLETE_ANALYSIS.md)
- [Performance Calculation Guide](../../PERF_CALCULATION_ANALYSIS.md)
- [D2M-JIT Pattern Framework](../../tools/d2m-jit/README.md)

---

*For issues or questions, see the main tt-mlir documentation or file an issue.*
