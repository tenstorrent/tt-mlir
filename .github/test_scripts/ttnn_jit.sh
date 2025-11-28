#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

export PYTHONPATH="$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal"

# Download and install ttnn-jit wheel
echo "Downloading ttnn-jit wheel..."
cd $WORK_DIR
gh run download $RUN_ID --repo tenstorrent/tt-mlir --name ttnn-jit-whl-$IMAGE_NAME

echo "Installing ttnn-jit wheel..."
if pip show ttnn-jit &> /dev/null; then
    pip uninstall -y ttnn-jit
fi
pip install ttnn_jit*.whl --upgrade

echo "Running ttnn-jit tests..."
pytest -v $WORK_DIR/test/ttnn-jit/ --junit-xml=$TEST_REPORT_PATH
