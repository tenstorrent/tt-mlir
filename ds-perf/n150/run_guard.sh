#!/usr/bin/env bash
# Device perf for the guard variant only, on the models the guard actually changed
# plus one 8B model it left alone as a control.
cd "$(dirname "$0")"
export DSPERF=/localdev/bmalesevic/tt-mlir/ds-perf
export GRAPHS=$PWD/graphs
export OUT=$PWD/fleet
export VARIANTS=guard
for m in falcon3_7b llama_3_1_8b mistral_7b qwen_3_8b qwen_2_5_7b; do
  [ -f "$OUT/${m}__guard.matmuls.csv" ] && { echo "### skip $m"; continue; }
  echo "### $m  $(date +%H:%M:%S)"
  MODELS="$m" bash ../run_model_fleet.sh 2>&1 | sed 's/^/    /'
done
echo "### GUARD DONE $(date +%H:%M:%S)"
