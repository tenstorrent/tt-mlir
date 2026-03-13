#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run each parametrized test in perf_tests.py under the device profiler (tracy)
# and dump results into a directory per test. Use TT_METAL_PROFILER_DIR so each
# run writes to a known subdir. At the end, runs summarize_perf_results.py to
# produce one JSON report per test case in OUT_DIR (perf_<op>_<dtype>_<mem>_<JOB_ID>.json).
# Set JOB_ID env var to include the job ID in filenames (required for CI).
#
# Usage:
#   ./test/ttnn-jit/perf_ci/run_perf_collect.sh [OUT_DIR]
#
# Example:
#   ./test/ttnn-jit/perf_ci/run_perf_collect.sh
#   ./test/ttnn-jit/perf_ci/run_perf_collect.sh generated/jit_perf_reports/my_run

set -e

# Script lives in test/ttnn-jit/perf_ci/; go up three levels to repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f env/activate ]; then
  # shellcheck source=/dev/null
  source env/activate
fi

OUT_DIR="${1:-generated/jit_perf_reports/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# Collect test ids from perf_tests.py (whatever is parametrized there)
collect_out=$(mktemp)
if ! pytest test/ttnn-jit/perf_ci/perf_tests.py --collect-only -q >"$collect_out" 2>&1; then
  echo "Error: pytest collect failed:" >&2
  cat "$collect_out" >&2
  rm -f "$collect_out"
  exit 1
fi
TESTS=($(sed -n 's/.*test_op_compare\[\(.*\)\]/\1/p' <"$collect_out"))
if [ ${#TESTS[@]} -eq 0 ]; then
  echo "Error: no test_op_compare[*] tests found in test/ttnn-jit/perf_ci/perf_tests.py. Pytest collect output:" >&2
  cat "$collect_out" >&2
  rm -f "$collect_out"
  exit 1
fi
rm -f "$collect_out"
echo "Collected ${#TESTS[@]} tests from perf_tests.py"

export TT_METAL_DEVICE_PROFILER=1

for tid in "${TESTS[@]}"; do
  echo "=============================================="
  echo "Running test_op_compare[$tid] ..."
  echo "=============================================="
  export TT_METAL_PROFILER_DIR="$OUT_DIR/$tid"
  mkdir -p "$TT_METAL_PROFILER_DIR"
  if ! python -m tracy -m -r -p "pytest test/ttnn-jit/perf_ci/perf_tests.py::test_op_compare[$tid]"; then
    echo "Warning: test_op_compare[$tid] exited with non-zero status (results may still be present)."
  fi
done

echo ""
echo "Results written under: $OUT_DIR"
echo "Summarizing..."
JOB_ID_ARG=""
if [ -n "$JOB_ID" ]; then
  JOB_ID_ARG="--job-id $JOB_ID"
fi
if python test/ttnn-jit/perf_ci/summarize_perf_results.py "$OUT_DIR" --output-dir "$OUT_DIR" $JOB_ID_ARG; then
  echo "Summary reports written to $OUT_DIR"
else
  echo "Warning: summarizer exited with an error (run dir may be partial)." >&2
fi
