#!/usr/bin/env bash
# Measure decode-graph (g1) device perf for a fleet of models, DS vs no-DS.
#
# Uses the CI artifacts' pre-lowered ttnn_runtime_*_g1_*.mlir, which translate to
# flatbuffer directly (no --ttnn-common-to-runtime-pipeline needed).
#
# Durations come from percore_perf.py, not ops_perf_results.csv: the cross-core
# clock epoch bug corrupts DEVICE KERNEL DURATION for every multi-core op.
# No `set -u` -- env/activate probes unset variables.

cd ${TTMLIR:-/localdev/bmalesevic/tt-mlir}
source env/activate

S=${DSPERF:-$(cd "$(dirname "$0")" && pwd)}          # where these scripts live
M=${GRAPHS:?set GRAPHS=/path/to/downloaded/graphs}
OUT=${OUT:-$PWD/fleet}
# ttrt prints a warning on stdout, so take the last line only.
PROF=$(python -c "import ttrt.runtime,os;print(os.path.dirname(ttrt.runtime.__file__))" 2>/dev/null | tail -1)/generated/profiler
LOGS=$PROF/.logs
export TT_METAL_PROFILER_DIR=$PROF
export TT_METAL_PROFILER_TRACE_TRACKING=1     # BEGIN/REPLAY markers; else post-processing dies
LOOPS=${LOOPS:-1}
MODELS=${MODELS:-"qwen_2_5_3b llama_3_2_1b qwen_2_5_1_5b qwen_3_0_6b falcon3_3b"}

mkdir -p $OUT

for m in $MODELS; do
  for v in ${VARIANTS:-ds nods}; do
    tag="${m}__${v}"
    src=$(find $M/$v/$m -name "ttnn_runtime_${m}*_g1_*.mlir" 2>/dev/null | head -1)
    if [ -z "$src" ]; then echo "=== $tag: NO g1 RUNTIME GRAPH ==="; continue; fi
    echo "=== $tag ==="
    fb=$OUT/$tag.ttnn
    if [ ! -f "$fb" ]; then
      if ! ttmlir-translate --ttnn-to-flatbuffer "$src" -o "$fb" 2>$OUT/$tag.translate.err; then
        echo "  TRANSLATE FAIL: $(grep -m1 -o 'error:.*' $OUT/$tag.translate.err | cut -c1-120)"
        continue
      fi
    fi
    ttrt perf "$fb" --loops $LOOPS --trace-region-size 268435456 \
        --enable-program-cache --ignore-version \
        --artifact-dir $OUT/$tag.art --log-file $OUT/$tag.ttrt.log \
        > $OUT/$tag.stdout 2>&1
    echo "  ttrt exit=$?"
    if [ ! -f "$LOGS/profile_log_device.csv" ]; then echo "  NO DEVICE LOG"; continue; fi

    python -O -c "
from tracy.process_ops_logs import process_ops
process_ops(None, None, False)" > $OUT/$tag.post.log 2>&1
    rc=$?
    python $S/percore_perf.py --device-log "$LOGS/profile_log_device.csv" \
          --ops-data "$LOGS/tracy_ops_data.csv" \
          --out $OUT/$tag.percore.csv --top 6 > $OUT/$tag.percore.log 2>&1
    # Per-matmul join needs the report; skip quietly if post-processing failed.
    rep=$PROF/reports/ops_perf_results.csv
    if [ $rc -eq 0 ] && [ -f "$rep" ]; then
      cp "$rep" $OUT/$tag.ops_perf_results.csv
      python $S/matmul_detail.py --percore $OUT/$tag.percore.csv --report "$rep" \
            --out $OUT/$tag.matmuls.csv > $OUT/$tag.matmuls.log 2>&1
    fi
    grep -E "^MatmulDeviceOperation|^total device" $OUT/$tag.percore.log | sed 's/^/  /'
  done
done
echo "FLEET DONE"
