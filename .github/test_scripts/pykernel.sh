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
if [ ! -d "$BUILD_DIR/python_packages/ttnn-jit" ]; then
    ln -sf tools/ttnn-jit $BUILD_DIR/python_packages/ttnn-jit
    ln -sf $BUILD_DIR/python_packages/ttrt/runtime $BUILD_DIR/python_packages/ttnn-jit/runtime
fi

echo "Running PyKernel tests..."
pytest -v $WORK_DIR/test/pykernel/demo/test.py $WORK_DIR/test/ttnn-jit/test_eltwise.py --junit-xml=$TEST_REPORT_PATH
rm -rf third_party/tt-metal
