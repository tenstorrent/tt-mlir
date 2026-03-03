#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

# The ttmlir and ttnn-jit wheels should already be built by the CMake flow.

# Find ttmlir wheel (built in python/dist/)
TTMLIR_WHEEL_PATH="$WORK_DIR/python/dist/ttmlir*.whl"
if ! ls $TTMLIR_WHEEL_PATH 1> /dev/null 2>&1; then
    echo "Error: ttmlir wheel not found at $TTMLIR_WHEEL_PATH"
    echo "Expected wheel to be built by CMake ttmlir-wheel target"
    exit 1
fi
echo "Found ttmlir wheel: $(ls $TTMLIR_WHEEL_PATH)"

# Find ttnn-jit wheel (built in tools/ttnn-jit/build/)
TTNN_JIT_WHEEL_PATH="$WORK_DIR/tools/ttnn-jit/build/ttnn_jit*.whl"
if ! ls $TTNN_JIT_WHEEL_PATH 1> /dev/null 2>&1; then
    echo "Error: ttnn-jit wheel not found at $TTNN_JIT_WHEEL_PATH"
    echo "Expected wheel to be built by CMake ttnn-jit target"
    exit 1
fi
echo "Found ttnn-jit wheel: $(ls $TTNN_JIT_WHEEL_PATH)"

# Upload artifacts
echo "{\"name\":\"ttmlir-whl-$BUILD_NAME\",\"path\":\"$TTMLIR_WHEEL_PATH\"}," >> $UPLOAD_LIST
echo "{\"name\":\"ttnn-jit-whl-$BUILD_NAME\",\"path\":\"$TTNN_JIT_WHEEL_PATH\"}," >> $UPLOAD_LIST
