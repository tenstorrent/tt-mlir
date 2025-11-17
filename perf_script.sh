#!/bin/bash

set -e

source env/activate
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_DIR=/localdev/brapanan/tt-mlir/generated/profiler_jit_muladd/
export TT_METAL_HOME=/localdev/brapanan/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH="${PYTHONPATH}:${TT_METAL_HOME}/ttnn:${TT_METAL_HOME}:${TT_METAL_HOME}/tools"

for test_params in "3410-640" "8190-1280" "16380-2560" "32760-5120" "37800-640" "75600-1280"; do
    python -m tracy -r -m pytest test/ttnn-jit/multiply_accumulate_trace.py::test_muladd_dram_trace[$test_params]
    python3 generated/extract_perf_columns.py -d $TT_METAL_PROFILER_DIR -o ${TT_METAL_PROFILER_DIR}/${test_params}_clean.csv
    python3 generated/extract_total_durations.py ${TT_METAL_PROFILER_DIR}/${test_params}_clean.csv -b 1
done
