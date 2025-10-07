#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Set up PYTHONPATH to include the built packages
export PYTHONPATH="$BUILD_DIR/python_packages:$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal:$PYTHONPATH"
mkdir -p $WORK_DIR/third_party/tt-metal
mkdir -p $WORK_DIR/third_party/tt-metal/src
ln -sf $INSTALL_DIR/tt-metal third_party/tt-metal/src/tt-metal

echo "Running PyKernel tests..."
pytest -v $WORK_DIR/test/pykernel/demo/test.py
pytest -v $WORK_DIR/test/ttnn-jit/test_eltwise.py
rm -rf third_party/tt-metal
