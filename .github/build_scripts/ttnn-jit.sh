#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

# The ttnn-jit wheel should already be built by the CMake flow.

# Find ttnn-jit wheel (built in tools/ttnn-jit/build/)
TTNN_JIT_WHEEL_PATH="$WORK_DIR/tools/ttnn-jit/build/ttnn_jit*.whl"
if ! ls $TTNN_JIT_WHEEL_PATH 1> /dev/null 2>&1; then
    echo "Error: ttnn-jit wheel not found at $TTNN_JIT_WHEEL_PATH"
    echo "Expected wheel to be built by CMake ttnn-jit target"
    exit 1
fi
echo "Found ttnn-jit wheel: $(ls $TTNN_JIT_WHEEL_PATH)"

# Upload artifact
echo "{\"name\":\"ttnn-jit-whl-$BUILD_NAME\",\"path\":\"$TTNN_JIT_WHEEL_PATH\"}," >> $UPLOAD_LIST
