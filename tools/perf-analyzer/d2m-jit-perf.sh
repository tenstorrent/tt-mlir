#!/usr/bin/env bash
set -euo pipefail

# Usage: d2m-jit-perf.sh <test-to-run> <traits-to-instrument> <output-dir>
if [ "$#" -ne 3 ]; then
  echo "usage: $0 <test-to-run> <traits-to-instrument> <output-dir>" >&2
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f env/activate ]; then
  source env/activate
fi

# DISPATCH=0 avoids the device-profiler hang. tracy -r sets
# TT_METAL_DEVICE_PROFILER=1 and -o sets TT_METAL_PROFILER_DIR for us.
export TT_METAL_DEVICE_PROFILER_DISPATCH=0
export D2M_JIT_ENABLE_PERF_TRACE=1
export D2M_JIT_INSERT_PROFILER_TRACES=1
export D2M_JIT_PROFILER_TRAITS="$2"

LOG_PATH=$(realpath -m "$3")

echo "running test $1 with instrumentation on $2"

# python -m tracy wraps the run under a Tracy capture server, then post-processes:
#   -r  generate report (starts capture-release, writes the .tracy file)
#   -m  treat the target as a module (runpy: equivalent to `python -m pytest`)
#   -o  profiler artifacts output folder (its .logs/ is wiped per run)
#   -v  verbose output
python -m tracy -r -m -v -o "$LOG_PATH" pytest "$1"

echo "tracy file      -> $LOG_PATH/.logs/tracy_profile_log_host.tracy"
echo "device perf csv -> $LOG_PATH/.logs/profile_log_device.csv"
echo "ops perf report -> $LOG_PATH/reports/<timestamp>/ops_perf_results_<timestamp>.csv"
