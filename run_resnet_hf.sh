#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e

echo "Running ResNet HF pipeline..."

# Step 1: Query and save artifacts
echo "Step 1: Running ttrt query..."
ttrt query --save-artifacts

# Step 2: Set system descriptor path
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
echo "SYSTEM_DESC_PATH=$SYSTEM_DESC_PATH"

# Step 3: Convert TTIR to TTNN
echo "Step 2: Converting TTIR to TTNN backend..."
ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=$SYSTEM_DESC_PATH enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" \
  -o resnet_hf_ttnn.mlir test/ttmlir/models/resnet_hf.mlir

# Step 4: Translate to flatbuffer
echo "Step 3: Translating to flatbuffer..."
ttmlir-translate --ttnn-to-flatbuffer -o resnet_hf.ttnn resnet_hf_ttnn.mlir

# Step 5: Run on device
echo "Step 4: Running on device..."
ttrt run resnet_hf.ttnn

echo "Pipeline completed successfully!"
