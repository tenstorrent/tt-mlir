# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -e

source env/activate
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_DIR=/localdev/brapanan/tt-mlir/third_party/tt-metal/src/tt-metal/generated/profiler/
export TT_METAL_HOME=/localdev/brapanan/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH="${PYTHONPATH}:${TT_METAL_HOME}/ttnn:${TT_METAL_HOME}:${TT_METAL_HOME}/tools"

python -m tracy -r -m pytest test/ttnn-jit/test_jit_resnet.py -x -svv
python3 third_party/tt-metal/src/tt-metal/generated/profiler/reports/extract_perf.py -d $TT_METAL_PROFILER_DIR -o ${TT_METAL_PROFILER_DIR}/${test_params}_resnet_jit_2.csv

python -m tracy -r -m pytest test/ttnn-jit/test_resnet.py -x -svv
python3 third_party/tt-metal/src/tt-metal/generated/profiler/reports/extract_perf.py -d $TT_METAL_PROFILER_DIR -o ${TT_METAL_PROFILER_DIR}/${test_params}_resnet_no_jit_2.csv
