#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run d2m-jit pattern perf benchmarks and summarize results for Superset.
#
# The conftest.py perf_runner fixture reads device profiler data after each
# test and writes a combined perf_results.json to OUT_DIR.
# summarize_perf_results.py then turns that into one JSON report per case.
#
# Usage:
#   ./test/d2m-jit/perf_ci/run_perf_collect.sh [OUT_DIR] [PYTEST_EXTRA_ARGS...]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "$VIRTUAL_ENV" ] && [ -f env/activate ]; then
  # shellcheck source=/dev/null
  source env/activate
fi

OUT_DIR="${1:-generated/d2m_perf_reports/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
shift || true

export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_MID_RUN_DUMP=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_PROFILER_DIR="$OUT_DIR"

REPORT_DIR="$OUT_DIR/reports"
mkdir -p "$REPORT_DIR"

PERF_CI_DIR="test/d2m-jit/perf_ci"

echo "Running d2m-jit perf benchmarks..."
if ! pytest \
  "$PERF_CI_DIR/perf_tests.py" \
  "$@"; then
  echo "Warning: pytest exited with non-zero status (results may still be present)."
fi

echo ""
echo "Summarizing..."
JOB_ID_ARG=""
if [ -n "$JOB_ID" ]; then
  JOB_ID_ARG="--job-id $JOB_ID"
fi
if python test/d2m-jit/perf_ci/summarize_perf_results.py "$OUT_DIR" --output-dir "$REPORT_DIR" $JOB_ID_ARG; then
  echo "Summary reports written to $REPORT_DIR"
  if [ -n "$UPLOAD_LIST" ]; then
    echo "{\"name\":\"d2m-perf-reports-${JOB_ID:-local}\",\"path\":\"$REPORT_DIR\"}," >> "$UPLOAD_LIST"
  fi
else
  echo "Warning: summarizer exited with an error (run dir may be partial)." >&2
fi
