#!/usr/bin/env bash
# Translate + profile every planned DS config; durations per core (epoch bug).
cd /localdev/bmalesevic/tt-mlir
source env/activate
T=/home/bmalesevic/.claude/jobs/30b23368/tmp
D=$T/sweep; R=$T/sweep/results; mkdir -p $R
PROF=/localdev/bmalesevic/tt-mlir/build/python_packages/ttrt/runtime/generated/profiler
LOGS=$PROF/.logs
export TT_METAL_PROFILER_DIR=$PROF
export TT_METAL_PROFILER_TRACE_TRACKING=1
PC=/tmp/claude-1211409153/-localdev-bmalesevic-tt-mlir/8c295604-d6ae-4366-b929-2c354e16bbe6/scratchpad/percore_perf.py

for m in $D/*.mlir; do
  n=$(basename $m .mlir)
  [ -f $R/$n.percore.csv ] && continue
  if [ ! -f $D/$n.ttnn ]; then
    if ! ttmlir-translate --ttnn-to-flatbuffer $m -o $D/$n.ttnn 2>$R/$n.terr; then
      echo "$n TRANSLATE_FAIL"; continue
    fi
  fi
  ttrt perf $D/$n.ttnn --loops 5 --ignore-version \
      --artifact-dir $R/$n.art --log-file $R/$n.log > $R/$n.stdout 2>&1
  rc=$?
  if [ ! -f "$LOGS/profile_log_device.csv" ]; then echo "$n NO_LOG (exit $rc)"; continue; fi
  python $PC --device-log "$LOGS/profile_log_device.csv" \
        --ops-data "$LOGS/tracy_ops_data.csv" --out $R/$n.percore.csv --top 3 \
        > $R/$n.percore.log 2>&1
  mm=$(awk -F, '$3=="MatmulDeviceOperation"{print $5}' $R/$n.percore.csv 2>/dev/null | sort -n | head -1)
  echo "$n  exit=$rc  matmul_ns=${mm:-NONE}"
  rm -f "$LOGS/profile_log_device.csv"
done
echo SWEEP_DONE
