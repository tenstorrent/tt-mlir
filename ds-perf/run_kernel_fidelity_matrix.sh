#!/usr/bin/env bash
# Profile every cell of the kernel x fidelity matrix, one config per process.
#
# One process per config so each device log holds a single matmul shape, and durations
# come from percore_perf.py -- DEVICE KERNEL DURATION is corrupt for multi-core ops on
# this card (see README.md).
cd "$(dirname "$0")/.."
source env/activate

R=${OUT:-ds-perf/results/kernel_fidelity}
mkdir -p "$R"
PROF=$PWD/build/python_packages/ttrt/runtime/generated/profiler
LOGS=$PROF/.logs
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_DIR=$PROF

CASES=$(python -c "
import sys; sys.path.insert(0,'ds-perf')
import test_kernel_fidelity_matrix as m
print(' '.join(c[0] for c in m.CASES))" 2>/dev/null)
[ -z "$CASES" ] && { echo "could not enumerate cases"; exit 1; }

for c in $CASES; do
  [ -f "$R/$c.percore.csv" ] && { echo "skip $c"; continue; }
  rm -f "$LOGS/profile_log_device.csv"
  pytest -s ds-perf/test_kernel_fidelity_matrix.py -k "$c" \
      > "$R/$c.stdout" 2>&1
  rc=$?
  if [ ! -f "$LOGS/profile_log_device.csv" ]; then
    echo "$c  NO_DEVICE_LOG (pytest exit $rc)"; continue
  fi
  python ds-perf/percore_perf.py --device-log "$LOGS/profile_log_device.csv" \
      --ops-data "$LOGS/tracy_ops_data.csv" --out "$R/$c.percore.csv" --top 3 \
      > "$R/$c.percore.log" 2>&1
  n=$(awk -F, '$3=="MatmulDeviceOperation"' "$R/$c.percore.csv" 2>/dev/null | wc -l)
  echo "$c  exit=$rc  matmul_rows=$n"
done
echo MATRIX_DONE
