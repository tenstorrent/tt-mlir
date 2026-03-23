#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Download alchemist wheel"
gh run download $RUN_ID --repo tenstorrent/tt-mlir --name "tt-alchemist-whl-speedy"

if type deactivate >/dev/null 2>&1; then
    deactivate
fi

python3 -m venv testenv
source testenv/bin/activate
pip install tt_alchemist-*.whl --force-reinstall
echo "Wheel installed successfully"
rm -rf tt_alchemist-*.whl

echo "Run tt-alchemist API test - model-to-cpp"
tt-alchemist model-to-cpp $WORK_DIR/tools/tt-alchemist/test/models/mnist.mlir

echo "Run tt-alchemist API test - model-to-python"
tt-alchemist model-to-python $WORK_DIR/tools/tt-alchemist/test/models/mnist.mlir

echo "Run tt-alchemist API test - generate-cpp (load tensors)"
OUTPUT_DIR=/tmp/test-load-tensors-cpp
rm -rf $OUTPUT_DIR
tt-alchemist generate-cpp --pipeline-options 'load-input-tensors-from-disk=true' $WORK_DIR/test/ttmlir/EmitC/TTNN/load_input/load_input_local/ttnn_load_input_tensors.mlir --output $OUTPUT_DIR --standalone
cp $WORK_DIR/test/ttmlir/EmitC/TTNN/load_input/load_input_local/arg0.tensorbin $WORK_DIR/test/ttmlir/EmitC/TTNN/load_input/load_input_local/arg1.tensorbin $OUTPUT_DIR
cd $OUTPUT_DIR
[ -d $OUTPUT_DIR ] || { echo "Directory not found: $OUTPUT_DIR" >&2; exit 1; }
./run

echo "Run tt-alchemist API test - generate-cpp (mnist)"
rm -rf /tmp/test-generate-cpp-mnist
tt-alchemist generate-cpp $WORK_DIR/tools/tt-alchemist/test/models/mnist.mlir --output /tmp/test-generate-cpp-mnist --standalone
cd /tmp/test-generate-cpp-mnist
[ -d /tmp/test-generate-cpp-mnist ] || { echo "Directory not found: /tmp/test-generate-cpp-mnist" >&2; exit 1; }
./run

echo "Run tt-alchemist API test - generate-cpp (resnet)"
rm -rf /tmp/test-generate-cpp-resnet
tt-alchemist generate-cpp $WORK_DIR/tools/tt-alchemist/test/models/resnet_hf.mlir --output /tmp/test-generate-cpp-resnet --standalone
cd /tmp/test-generate-cpp-resnet
[ -d /tmp/test-generate-cpp-resnet ] || { echo "Directory not found: /tmp/test-generate-cpp-resnet" >&2; exit 1; }
./run

echo "Run tt-alchemist API test - generate-python (load tensors)"
OUTPUT_DIR=/tmp/test-load-tensors-python
rm -rf $OUTPUT_DIR
tt-alchemist generate-python --pipeline-options 'load-input-tensors-from-disk=true' $WORK_DIR/test/ttmlir/EmitPy/TTNN/load_input/load_input_local/ttnn_load_input_tensors.mlir --output $OUTPUT_DIR
cp $WORK_DIR/test/ttmlir/EmitPy/TTNN/load_input/load_input_local/arg0.tensorbin $WORK_DIR/test/ttmlir/EmitPy/TTNN/load_input/load_input_local/arg1.tensorbin $OUTPUT_DIR
cd $OUTPUT_DIR
[ -d $OUTPUT_DIR ] || { echo "Directory not found: $OUTPUT_DIR" >&2; exit 1; }
# ./run  # TODO: enable when fixed

echo "Run tt-alchemist API test - generate-python (mnist)"
rm -rf /tmp/test-generate-python
tt-alchemist generate-python $WORK_DIR/tools/tt-alchemist/test/models/mnist.mlir --output /tmp/test-generate-python --standalone
cd /tmp/test-generate-python
[ -d /tmp/test-generate-python ] || { echo "Directory not found: /tmp/test-generate-python" >&2; exit 1; }
# ./run  # TODO: enable when fixed

echo "Run tt-alchemist API test - generate-python (TTNN input)"
# First convert TTIR to TTNN using ttmlir-opt
TTNN_MLIR=/tmp/test-ttnn-input.mlir
ttmlir-opt --ttir-to-ttnn-backend-pipeline $WORK_DIR/tools/tt-alchemist/test/models/add.mlir -o $TTNN_MLIR
# Then run tt-alchemist on the TTNN input
rm -rf /tmp/test-generate-python-ttnn
tt-alchemist generate-python $TTNN_MLIR --output /tmp/test-generate-python-ttnn
cd /tmp/test-generate-python-ttnn
[ -d /tmp/test-generate-python-ttnn ] || { echo "Directory not found: /tmp/test-generate-python-ttnn" >&2; exit 1; }
# ./run  # TODO: enable when fixed

echo "Test Passed. Doing cleanup"
deactivate
rm -rf testenv
cd $WORK_DIR
source env/activate

echo "Run C++ python_runner tests"
export TT_METAL_HOME="$WORK_DIR/third_party/tt-metal/src/tt-metal"
export TT_METAL_RUNTIME_ROOT="$INSTALL_DIR/tt-metal"
export TT_METAL_LIB="$INSTALL_DIR/lib"
export LD_LIBRARY_PATH="$INSTALL_DIR/tools/tt-alchemist/test:$INSTALL_DIR/lib:$INSTALL_DIR/tt-metal/lib:${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH}"
# Add paths to PYTHONPATH: test directory for test_model.py, ttnn and tt-metal for the ttnn module
export PYTHONPATH="$INSTALL_DIR/tools/tt-alchemist/test:$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal:${PYTHONPATH:-}"
cd "$INSTALL_DIR/tools/tt-alchemist/test"

echo "Run test_python_runner_simple"
./test_python_runner_simple

echo "Run test_python_runner (requires device)"
./test_python_runner

cd $WORK_DIR
