#!/usr/bin/env bash

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Set strict mode.
set -euo pipefail

# Usage: d2m-jit-perf.sh <test-to-run> <traits-to-instrument> <output-dir>
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <test-to-run> <traits-to-instrument> <output-dir> [system-descriptor]" >&2
  exit 1
fi

# Check if the user is in a venv.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f env/activate ]; then
  # env/activate references unset vars; disable nounset while sourcing it.
  set +u
  # shellcheck source=/dev/null
  source env/activate
  set -u
fi

# Export some necessary environment variables.
# DISPATCH=0 avoids the device-profiler hang. tracy -r sets
# TT_METAL_DEVICE_PROFILER=1 and -o sets TT_METAL_PROFILER_DIR for us.
export TT_METAL_DEVICE_PROFILER_DISPATCH=0
export D2M_JIT_ENABLE_PERF_TRACE=1
export D2M_JIT_INSERT_PROFILER_TRACES=1
export D2M_JIT_PROFILER_TRAITS="$2"
if [ "$#" -eq 4 ]; then
  export SYSTEM_DESC_PATH="$4"
fi

LOG_PATH=$(realpath -m "$3")

echo "running test $1 with instrumentation on $2..."

echo "cleaning up previous logs at $LOG_PATH/.logs"
rm -rf "$LOG_PATH/.logs"

# python -m tracy wraps the run under a Tracy capture server, then post-processes:
#   -r  generate report (starts tracy-capture, writes the .tracy file)
#   -m  treat the target as a module (runpy: equivalent to `python -m pytest`)
#   -o  profiler artifacts output folder (its .logs/ is wiped per run)
#   -v  verbose output
python -m tracy -r -v -o "$LOG_PATH" --profiler-capture-perf-counters=all -m pytest "$1"

echo "tracy file      -> $LOG_PATH/reports/<timestamp>/tracy_profile_log_host.tracy"
echo "ops perf report -> $LOG_PATH/reports/<timestamp>/ops_perf_results_<timestamp>.csv"
echo "device perf csv -> $LOG_PATH/reports/<timestamp>/profile_log_device.csv"
