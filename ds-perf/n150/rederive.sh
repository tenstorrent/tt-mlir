#!/usr/bin/env bash
# Re-join per-matmul CSVs from the saved profiler artifacts (no device time needed).
# Run after matmul_detail.py changes, or to pick up the fused-activation column.
set -u
PY=/opt/ttmlir-toolchain/venv/bin/python
for pc in fleet/*.percore.csv; do
  tag=$(basename "$pc" .percore.csv)
  rep="fleet/$tag.ops_perf_results.csv"
  [ -f "$rep" ] || { echo "skip $tag (no report)"; continue; }
  $PY ../matmul_detail.py --percore "$pc" --report "$rep" \
      --out "fleet/$tag.matmuls.csv" > "fleet/$tag.matmuls.log" 2>&1 \
    && echo "ok $tag" || echo "FAIL $tag"
done
