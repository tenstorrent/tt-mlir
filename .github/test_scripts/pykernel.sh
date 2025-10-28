#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

# Set up PYTHONPATH to include the built packages
export PYTHONPATH="$BUILD_DIR/python_packages:$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal:$PYTHONPATH"
mkdir -p $WORK_DIR/third_party/tt-metal
mkdir -p $WORK_DIR/third_party/tt-metal/src
ln -sf $INSTALL_DIR/tt-metal third_party/tt-metal/src/tt-metal
if [ ! -d "$BUILD_DIR/python_packages/ttnn_jit" ]; then
    ln -sf $WORK_DIR/tools/ttnn-jit $BUILD_DIR/python_packages/ttnn_jit
fi
if [ ! -d "$BUILD_DIR/python_packages/ttnn_jit/runtime" ]; then
    ln -sf $BUILD_DIR/python_packages/ttrt/runtime $BUILD_DIR/python_packages/ttnn_jit/runtime
    mv $BUILD_DIR/python_packages/ttnn_jit/runtime/__init__.py $BUILD_DIR/python_packages/ttnn_jit/runtime/.__init__
fi

echo "Running PyKernel/ttnn-jit tests..."
pytest -v $WORK_DIR/test/pykernel/demo/test.py $WORK_DIR/test/ttnn-jit/ --junit-xml=$TEST_REPORT_PATH

# cleanup
rm -rf third_party/tt-metal
mv $BUILD_DIR/python_packages/ttnn_jit/runtime/.__init__ $BUILD_DIR/python_packages/ttnn_jit/runtime/__init__.py || true
