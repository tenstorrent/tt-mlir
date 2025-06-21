#!/bin/bash

# Debug Gather Test Script
# Issue #3849: CPU Fallback Gather Op Crash

set -e  # Exit on error

echo "=========================================="
echo "TT-MLIR Gather Operation Debug Session"
echo "Issue #3849: CPU Fallback Gather Op Crash"
echo "=========================================="

cd /home/linux/github/tt-mlir/build-minimal

echo "Step 1: Testing TTIR gather operation parsing..."
./bin/ttmlir-opt ../debug_gather.mlir --verify-diagnostics

echo -e "\nStep 2: Testing TTIR to TTIR decomposition..."
./bin/ttmlir-opt ../debug_gather.mlir --ttir-to-ttir-decomposition

echo -e "\nStep 3: Testing TTIR to Linalg conversion..."
./bin/ttmlir-opt ../debug_gather.mlir --convert-ttir-to-linalg

echo -e "\nStep 4: Testing full pipeline with minimal flags..."
./bin/ttmlir-opt ../debug_gather.mlir \
  --ttir-to-ttir-decomposition \
  --convert-ttir-to-linalg \
  --verify-diagnostics

echo -e "\n=========================================="
echo "Debug session completed successfully!"
echo "If any step failed, it indicates the crash location."
echo "=========================================="
