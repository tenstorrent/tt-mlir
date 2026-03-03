#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

export PYTHONPATH="$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal"

# Download and install ttmlir and ttnn-jit wheels
echo "Downloading wheels..."
cd $WORK_DIR

# This script may be called multiple times in the same test job.
# Delete the downloaded wheels since gh run will not overwrite an existing file.
if [ -f ttmlir*.whl ]; then
    rm -f ttmlir*.whl
fi
if [ -f ttnn_jit*.whl ]; then
    rm -f ttnn_jit*.whl
fi

# Download both wheels
gh run download $RUN_ID --repo tenstorrent/tt-mlir --name ttmlir-whl-$IMAGE_NAME
gh run download $RUN_ID --repo tenstorrent/tt-mlir --name ttnn-jit-whl-$IMAGE_NAME

echo "Installing ttmlir wheel..."
if pip show ttmlir &> /dev/null; then
    pip uninstall -y ttmlir
fi

echo "Installing ttnn-jit wheel..."
if pip show ttnn-jit &> /dev/null; then
    pip uninstall -y ttnn-jit
fi
# Use --find-links to ensure pip can find the ttmlir wheel
pip install ttnn_jit*.whl --find-links . --upgrade

echo "Running ttnn-jit tests..."
if [ "$1" == "nightly" ]; then
    # Run tests that are exclusive to the nightly workflow
    pytest -v $WORK_DIR/test/ttnn-jit/nightly/ --junit-xml=$TEST_REPORT_PATH
else
    if [[ "$RUNS_ON" == "n300-llmbox" ]]; then
        # only run multichip tests and matmul smoketests for llmbox
        pytest -v $WORK_DIR/test/ttnn-jit/test_mesh_tensor_eltwise.py $WORK_DIR/test/ttnn-jit/test_matmul_smoketest.py --junit-xml=$TEST_REPORT_PATH
    else
        # Only run tests in the top level directory. These are always run.
        llvm-lit -v --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttnn-jit/lit
        pytest -v $WORK_DIR/test/ttnn-jit/*.py --junit-xml=$TEST_REPORT_PATH
    fi
fi
