#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <shlo_file.mlir>"
  exit 1
fi

# Input file
SHLO_FILE="$1"

# Intermediate and output files
TTIR_FILE="ttir_file.mlir"
TTIR_FUSED_FILE="ttir_fused.mlir"
TTNN_FILE="ttnn_file.mlir"
FB_FILE="fb.ttnn"
LOG_FILE="run.log"

# Step 1: Convert SHLO to TTIR
echo "Running ttmlir-opt --stablehlo-to-ttir-pipeline..."
ttmlir-opt --stablehlo-to-ttir-pipeline "$SHLO_FILE" -o "$TTIR_FILE"
if [ $? -ne 0 ]; then
  echo "Error: Failed to convert SHLO to TTIR."
  exit 1
fi

# Step 2: Apply TTIR fusing
echo "Running ttmlir-opt --ttir-fusing..."
ttmlir-opt --ttir-fusing "$TTIR_FILE" -o "$TTIR_FUSED_FILE"
if [ $? -ne 0 ]; then
  echo "Error: Failed to apply TTIR fusing."
  exit 1
fi

# Step 3: Convert TTIR to TTNN backend
echo "Running ttmlir-opt --ttir-to-ttnn-backend-pipeline..."
ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=/localdev/jameszianxu/shlo/tt-mlir/ttrt-artifacts/system_desc.ttsys" "$TTIR_FUSED_FILE" -o "$TTNN_FILE"
if [ $? -ne 0 ]; then
  echo "Error: Failed to convert TTIR to TTNN backend."
  exit 1
fi

# Step 4: Translate TTNN to FlatBuffer
echo "Running ttmlir-translate --ttnn-to-flatbuffer..."
ttmlir-translate --ttnn-to-flatbuffer "$TTNN_FILE" -o "$FB_FILE"
if [ $? -ne 0 ]; then
  echo "Error: Failed to translate TTNN to FlatBuffer."
  exit 1
fi

# Step 5: Run TTRT and log results
echo "Running ttrt run..."
TTRT_LOGGER_LEVEL=DEBUG ttrt run "$FB_FILE" |& tee "$LOG_FILE"
if [ $? -ne 0 ]; then
  echo "Error: TTRT run failed. Check $LOG_FILE for details."
  exit 1
fi

echo "Pipeline completed successfully. Logs saved to $LOG_FILE."