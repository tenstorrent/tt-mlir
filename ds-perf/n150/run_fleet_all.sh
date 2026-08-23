#!/usr/bin/env bash
# Full n150 fleet: decode-graph device perf, DS vs no-DS, highest-value models first.
cd "$(dirname "$0")"
export DSPERF=/localdev/bmalesevic/tt-mlir/ds-perf
export GRAPHS=$PWD/graphs
export OUT=$PWD/fleet
MODELS_ORDERED="qwen_2_5_3b qwen_3_0_6b qwen_3_4b falcon3_1b llama_3_2_3b falcon3_3b \
qwen_3_1_7b llama_3_2_1b qwen_2_5_0_5b qwen_2_5_1_5b \
llama_3_1_8b mistral_7b qwen_2_5_7b qwen_3_8b falcon3_7b ministral_8b \
phi2 phi1 phi1_5 gemma_1_1_2b"
for m in $MODELS_ORDERED; do
  if [ -f "$OUT/${m}__ds.matmuls.csv" ] && [ -f "$OUT/${m}__nods.matmuls.csv" ]; then
    echo "### skip $m (already have both)"; continue
  fi
  echo "### $m  $(date +%H:%M:%S)"
  MODELS="$m" bash ../run_model_fleet.sh 2>&1 | sed 's/^/    /'
done
echo "### ALL DONE $(date +%H:%M:%S)"
