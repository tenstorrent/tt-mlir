#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Download alchemist wheel"
gh run download $RUN_ID --repo tenstorrent/tt-mlir --name "tt-alchemist-whl-speedy"

deactivate
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
OUTPUT_DIR=/tmp/test-load-tensors
rm -rf $OUTPUT_DIR
tt-alchemist generate-cpp --pipeline-options 'load-input-tensors-from-disk=true' $WORK_DIR/test/ttmlir/EmitC/TTNN/load_input/ttnn_load_input_tensors.mlir --output $OUTPUT_DIR --standalone
cp $WORK_DIR/test/ttmlir/EmitC/TTNN/load_input/arg0.tensorbin $WORK_DIR/test/ttmlir/EmitC/TTNN/load_input/arg1.tensorbin $OUTPUT_DIR
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

echo "Run tt-alchemist API test - generate-python"
rm -rf /tmp/test-generate-python
tt-alchemist generate-python $WORK_DIR/tools/tt-alchemist/test/models/mnist.mlir --output /tmp/test-generate-python --standalone
cd /tmp/test-generate-python
[ -d /tmp/test-generate-python ] || { echo "Directory not found: /tmp/test-generate-python" >&2; exit 1; }
# ./run  # TODO: enable when fixed

echo "Test Passed. Doing cleanup"
deactivate
rm -rf testenv
source env/activate
