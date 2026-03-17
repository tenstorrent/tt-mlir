#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run all parametrized tests in perf_tests.py in a single pytest session.
# The conftest.py perf_device fixture reads device profiler data after each
# test and writes a combined perf_results.json to OUT_DIR.
# summarize_perf_results.py then turns that into one JSON report per test case.
#
# Set JOB_ID env var to include the job ID in filenames (required for CI).
#
# Usage:
#   ./test/ttnn-jit/perf_ci/run_perf_collect.sh [OUT_DIR] [PYTEST_EXTRA_ARGS...]
#
# Examples:
#   ./test/ttnn-jit/perf_ci/run_perf_collect.sh
#   ./test/ttnn-jit/perf_ci/run_perf_collect.sh /tmp/perf_out --junit-xml=report.xml

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
shift || true

export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_MID_RUN_DUMP=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_PROFILER_DIR="$OUT_DIR"

echo "Running all perf tests..."
if ! pytest test/ttnn-jit/perf_ci/perf_tests.py "$@"; then
  echo "Warning: pytest exited with non-zero status (results may still be present)."
fi

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
