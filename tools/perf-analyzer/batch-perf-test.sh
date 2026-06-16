#!/usr/bin/env bash

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Batch performance testing script
# Collects tests from a pytest file, runs each through d2m-jit-perf.sh,
# and generates a comparative performance report.

set -euo pipefail

# Usage
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  cat >&2 <<EOF
usage: $0 <pytest-file> <output-base-dir> [traits] [system-descriptor]

  pytest-file       : Path to pytest file (e.g., test/d2m-jit/test_pattern_eltwise.py)
  output-base-dir   : Base directory for all profiling outputs
  traits            : Profiler traits (default: device-zone)
  system-descriptor : System descriptor path (default: ttrt-artifacts/system_desc.ttsys)

Example:
  $0 test/d2m-jit/test_pattern_eltwise.py prof_batch device-zone

This will:
  1. Discover all tests in the pytest file
  2. Run each test through d2m-jit-perf.sh
  3. Collect and analyze profiling data
  4. Generate a comparative performance report
EOF
  exit 1
fi

PYTEST_FILE="$1"
OUTPUT_BASE_DIR="$2"
TRAITS="${3:-device-zone}"
SYSTEM_DESC="${4:-ttrt-artifacts/system_desc.ttsys}"

# Check if pytest file exists
if [ ! -f "$PYTEST_FILE" ]; then
  echo "Error: pytest file '$PYTEST_FILE' not found" >&2
  exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if d2m-jit-perf.sh exists
PERF_SCRIPT="$SCRIPT_DIR/d2m-jit-perf.sh"
if [ ! -f "$PERF_SCRIPT" ]; then
  echo "Error: d2m-jit-perf.sh not found at $PERF_SCRIPT" >&2
  exit 1
fi

# Ensure we're in a venv
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/env/activate" ]; then
  set +u
  # shellcheck source=/dev/null
  source "$REPO_ROOT/env/activate"
  set -u
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"
OUTPUT_BASE_DIR="$(realpath "$OUTPUT_BASE_DIR")"

# Log file for the batch run
BATCH_LOG="$OUTPUT_BASE_DIR/batch_run.log"
SUMMARY_CSV="$OUTPUT_BASE_DIR/performance_summary.csv"
SUMMARY_REPORT="$OUTPUT_BASE_DIR/performance_report.md"

echo "==================================================================" | tee "$BATCH_LOG"
echo "Batch Performance Testing" | tee -a "$BATCH_LOG"
echo "==================================================================" | tee -a "$BATCH_LOG"
echo "Pytest file    : $PYTEST_FILE" | tee -a "$BATCH_LOG"
echo "Output dir     : $OUTPUT_BASE_DIR" | tee -a "$BATCH_LOG"
echo "Traits         : $TRAITS" | tee -a "$BATCH_LOG"
echo "System desc    : $SYSTEM_DESC" | tee -a "$BATCH_LOG"
echo "==================================================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"

# Step 1: Discover all tests in the pytest file
echo "📋 Discovering tests in $PYTEST_FILE..." | tee -a "$BATCH_LOG"
cd "$REPO_ROOT"

# Use pytest --collect-only to get test names
TESTS=$(python -m pytest "$PYTEST_FILE" --collect-only -q 2>/dev/null | grep '::' | grep -v '==' | grep -v 'session starts' || true)

if [ -z "$TESTS" ]; then
  echo "Error: No tests found in $PYTEST_FILE" >&2
  exit 1
fi

# Count tests
TEST_COUNT=$(echo "$TESTS" | wc -l)
echo "   Found $TEST_COUNT tests" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"

# Initialize CSV header
echo "test_name,total_device_time_ns,kernel_count,avg_kernel_time_ns,max_kernel_time_ns,min_kernel_time_ns,status,output_dir" > "$SUMMARY_CSV"

# Step 2: Run each test through d2m-jit-perf.sh
TEST_NUM=0
PASSED_COUNT=0
FAILED_COUNT=0

while IFS= read -r TEST_NAME; do
  TEST_NUM=$((TEST_NUM + 1))

  echo "==================================================================" | tee -a "$BATCH_LOG"
  echo "[$TEST_NUM/$TEST_COUNT] Running: $TEST_NAME" | tee -a "$BATCH_LOG"
  echo "==================================================================" | tee -a "$BATCH_LOG"

  # Create sanitized test directory name
  TEST_DIR_NAME=$(echo "$TEST_NAME" | sed 's/::/__/g' | sed 's/\[/_/g' | sed 's/\]//g' | sed 's/\//_/g')
  TEST_OUTPUT_DIR="$OUTPUT_BASE_DIR/$TEST_DIR_NAME"
  mkdir -p "$TEST_OUTPUT_DIR"

  # Run the performance test
  START_TIME=$(date +%s)
  if "$PERF_SCRIPT" "$TEST_NAME" "$TRAITS" "$TEST_OUTPUT_DIR" "$SYSTEM_DESC" >> "$BATCH_LOG" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    STATUS="PASSED"
    PASSED_COUNT=$((PASSED_COUNT + 1))
    echo "   ✅ Test passed (${ELAPSED}s)" | tee -a "$BATCH_LOG"
  else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    STATUS="FAILED"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    echo "   ❌ Test failed (${ELAPSED}s)" | tee -a "$BATCH_LOG"
  fi

  # Step 3: Analyze profiling data for this test
  echo "   📊 Analyzing profiling data..." | tee -a "$BATCH_LOG"

  # Find the most recent report directory
  REPORTS_DIR="$TEST_OUTPUT_DIR/reports"
  if [ -d "$REPORTS_DIR" ]; then
    LATEST_REPORT=$(ls -t "$REPORTS_DIR" | head -n1)
    if [ -n "$LATEST_REPORT" ]; then
      REPORT_PATH="$REPORTS_DIR/$LATEST_REPORT"

      # Look for ops_perf_results_*.csv or cpp_device_perf_report.csv
      DEVICE_PERF_CSV=$(find "$REPORT_PATH" -name "ops_perf_results_*.csv" -o -name "cpp_device_perf_report.csv" | head -n1)

      if [ -n "$DEVICE_PERF_CSV" ] && [ -f "$DEVICE_PERF_CSV" ]; then
        # Parse the performance data using Python
        PERF_DATA=$(python3 -c "
import csv
import sys

try:
    with open('$DEVICE_PERF_CSV', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print('0,0,0,0,0')
        sys.exit(0)

    # Extract kernel durations (in nanoseconds, convert to milliseconds)
    # Try multiple column names for compatibility
    durations_ns = []
    for row in rows:
        # Try ops_perf_results format first
        if 'DEVICE KERNEL DURATION [ns]' in row:
            try:
                dur_str = row['DEVICE KERNEL DURATION [ns]']
                if dur_str and dur_str.strip():
                    dur = float(dur_str)
                    durations_ns.append(dur)
            except (ValueError, KeyError):
                pass
        # Fall back to cpp_device_perf_report format
        elif 'KERNEL_DURATION' in row:
            try:
                dur = float(row['KERNEL_DURATION'])
                durations_ns.append(dur)
            except (ValueError, KeyError):
                pass

    if not durations_ns:
        print('0,0,0,0,0')
        sys.exit(0)

    # Keep values in nanoseconds (no conversion)
    total_ns = sum(durations_ns)
    count = len(durations_ns)
    avg_ns = total_ns / count
    max_ns = max(durations_ns)
    min_ns = min(durations_ns)

    print(f'{total_ns:.0f},{count},{avg_ns:.0f},{max_ns:.0f},{min_ns:.0f}')
except Exception as e:
    print(f'0,0,0,0,0', file=sys.stderr)
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")

        if [ -n "$PERF_DATA" ]; then
          IFS=',' read -r TOTAL_TIME KERNEL_COUNT AVG_TIME MAX_TIME MIN_TIME <<< "$PERF_DATA"
          echo "      Total device time: ${TOTAL_TIME}ns" | tee -a "$BATCH_LOG"
          echo "      Kernel count: $KERNEL_COUNT" | tee -a "$BATCH_LOG"
          echo "      Avg kernel time: ${AVG_TIME}ns" | tee -a "$BATCH_LOG"
          echo "      Max kernel time: ${MAX_TIME}ns" | tee -a "$BATCH_LOG"
          echo "      Min kernel time: ${MIN_TIME}ns" | tee -a "$BATCH_LOG"

          # Add to CSV
          echo "\"$TEST_NAME\",$TOTAL_TIME,$KERNEL_COUNT,$AVG_TIME,$MAX_TIME,$MIN_TIME,$STATUS,\"$TEST_OUTPUT_DIR\"" >> "$SUMMARY_CSV"
        else
          echo "      ⚠️  Could not parse performance data" | tee -a "$BATCH_LOG"
          echo "\"$TEST_NAME\",0,0,0,0,0,$STATUS,\"$TEST_OUTPUT_DIR\"" >> "$SUMMARY_CSV"
        fi
      else
        echo "      ⚠️  No performance CSV found (ops_perf_results_*.csv or cpp_device_perf_report.csv)" | tee -a "$BATCH_LOG"
        echo "\"$TEST_NAME\",0,0,0,0,0,$STATUS,\"$TEST_OUTPUT_DIR\"" >> "$SUMMARY_CSV"
      fi
    else
      echo "      ⚠️  No report directory found" | tee -a "$BATCH_LOG"
      echo "\"$TEST_NAME\",0,0,0,0,0,$STATUS,\"$TEST_OUTPUT_DIR\"" >> "$SUMMARY_CSV"
    fi
  else
    echo "      ⚠️  Reports directory not found" | tee -a "$BATCH_LOG"
    echo "\"$TEST_NAME\",0,0,0,0,0,$STATUS,\"$TEST_OUTPUT_DIR\"" >> "$SUMMARY_CSV"
  fi

  echo "" | tee -a "$BATCH_LOG"
done <<< "$TESTS"

# Step 4: Generate final summary report
echo "==================================================================" | tee -a "$BATCH_LOG"
echo "Generating Performance Report" | tee -a "$BATCH_LOG"
echo "==================================================================" | tee -a "$BATCH_LOG"

python3 - <<EOF | tee "$SUMMARY_REPORT"
import csv
import sys
from pathlib import Path

csv_file = Path("$SUMMARY_CSV")
if not csv_file.exists():
    print("Error: Summary CSV not found")
    sys.exit(1)

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print("No test data collected")
    sys.exit(0)

print("# Performance Test Report")
print()
print(f"**Pytest File:** \`$PYTEST_FILE\`")
print(f"**Output Directory:** \`$OUTPUT_BASE_DIR\`")
print(f"**Profiler Traits:** \`$TRAITS\`")
print(f"**Total Tests:** $TEST_COUNT")
print(f"**Passed:** $PASSED_COUNT")
print(f"**Failed:** $FAILED_COUNT")
print()

print("## Summary Statistics")
print()

# Calculate aggregate stats
passed_rows = [r for r in rows if r['status'] == 'PASSED']
if passed_rows:
    total_times_ns = [float(r['total_device_time_ns']) for r in passed_rows if float(r['total_device_time_ns']) > 0]
    kernel_counts = [int(r['kernel_count']) for r in passed_rows if int(r['kernel_count']) > 0]
    avg_times_ns = [float(r['avg_kernel_time_ns']) for r in passed_rows if float(r['avg_kernel_time_ns']) > 0]

    if total_times_ns:
        # Convert ns to ms for display
        print(f"- **Total Device Time (all tests):** {sum(total_times_ns)/1e6:.3f}ms")
        print(f"- **Average Test Time:** {sum(total_times_ns)/len(total_times_ns)/1e6:.3f}ms")
        print(f"- **Longest Test:** {max(total_times_ns)/1e6:.3f}ms")
        print(f"- **Shortest Test:** {min(total_times_ns)/1e6:.3f}ms")
    if kernel_counts:
        print(f"- **Total Kernels:** {sum(kernel_counts)}")
        print(f"- **Average Kernels per Test:** {sum(kernel_counts)/len(kernel_counts):.1f}")
    if avg_times_ns:
        print(f"- **Average Kernel Time (across all):** {sum(avg_times_ns)/len(avg_times_ns)/1e6:.3f}ms")
else:
    print("No successful tests with performance data.")

print()
print("## Detailed Results")
print()
print("| Test Name | Status | Total Time (ns) | Kernels | Avg Kernel (ns) | Max Kernel (ns) | Min Kernel (ns) |")
print("|-----------|--------|-----------------|---------|-----------------|-----------------|-----------------|")

for row in rows:
    test_name = row['test_name'].split('::')[-1]  # Get just the test function name
    if len(test_name) > 50:
        test_name = test_name[:47] + "..."

    status_icon = "✅" if row['status'] == "PASSED" else "❌"
    total = f"{float(row['total_device_time_ns']):.0f}" if float(row['total_device_time_ns']) > 0 else "N/A"
    kernels = row['kernel_count'] if int(row['kernel_count']) > 0 else "N/A"
    avg = f"{float(row['avg_kernel_time_ns']):.0f}" if float(row['avg_kernel_time_ns']) > 0 else "N/A"
    max_k = f"{float(row['max_kernel_time_ns']):.0f}" if float(row['max_kernel_time_ns']) > 0 else "N/A"
    min_k = f"{float(row['min_kernel_time_ns']):.0f}" if float(row['min_kernel_time_ns']) > 0 else "N/A"

    print(f"| {test_name} | {status_icon} | {total} | {kernels} | {avg} | {max_k} | {min_k} |")

print()
print("## Performance Ranking (by Total Device Time)")
print()

# Sort by total device time
passed_with_data = [r for r in passed_rows if float(r['total_device_time_ns']) > 0]
passed_with_data.sort(key=lambda x: float(x['total_device_time_ns']), reverse=True)

if passed_with_data:
    print("| Rank | Test Name | Total Time (ns) | Kernels |")
    print("|------|-----------|-----------------|---------|")
    for i, row in enumerate(passed_with_data[:10], 1):  # Top 10
        test_name = row['test_name'].split('::')[-1]
        if len(test_name) > 60:
            test_name = test_name[:57] + "..."
        total = f"{float(row['total_device_time_ns']):.0f}"
        kernels = row['kernel_count']
        print(f"| {i} | {test_name} | {total} | {kernels} |")

    if len(passed_with_data) > 10:
        print(f"|  | *({len(passed_with_data) - 10} more tests not shown)* | | |")
else:
    print("No performance data available.")

print()
print("## Output Locations")
print()
print("- **Summary CSV:** \`$SUMMARY_CSV\`")
print("- **Batch Log:** \`$BATCH_LOG\`")
print("- **Individual Test Data:** \`$OUTPUT_BASE_DIR/<test_name>/\`")
print()
print("---")
print()
print("*Generated by batch-perf-test.sh*")

EOF

echo "" | tee -a "$BATCH_LOG"
echo "==================================================================" | tee -a "$BATCH_LOG"
echo "Batch Run Complete!" | tee -a "$BATCH_LOG"
echo "==================================================================" | tee -a "$BATCH_LOG"
echo "Summary CSV    : $SUMMARY_CSV" | tee -a "$BATCH_LOG"
echo "Summary Report : $SUMMARY_REPORT" | tee -a "$BATCH_LOG"
echo "Batch Log      : $BATCH_LOG" | tee -a "$BATCH_LOG"
echo "==================================================================" | tee -a "$BATCH_LOG"
echo ""
echo "Tests completed: $TEST_COUNT"
echo "  Passed: $PASSED_COUNT"
echo "  Failed: $FAILED_COUNT"
echo ""
echo "📊 View the report:"
echo "   cat $SUMMARY_REPORT"

exit 0
