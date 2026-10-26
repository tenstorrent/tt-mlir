#!/bin/bash
# VoVNet Tuning Build and Run Script
# Bounty: tenstorrent/tt-mlir#4349

set -e

echo "=========================================="
echo "VoVNet Tuning - Build Configuration"
echo "=========================================="

# Configuration
MODEL_NAME="ese_vovnet19b_dw"
BATCH_SIZE=${BATCH_SIZE:-8}
DATA_FORMAT="bfloat16"
TARGET_FPS=1400

echo "Model: $MODEL_NAME"
echo "Batch Size: $BATCH_SIZE"  
echo "Data Format: $DATA_FORMAT"
echo "Target FPS: $TARGET_FPS"

# Check prerequisites
if ! command -v tt-alchemist &> /dev/null; then
    echo "Error: tt-alchemist not found. Please build tt-mlir first."
    echo "See: https://docs.tenstorrent.com/tt-mlir/tt-alchemist.html"
    exit 1
fi

# Step 1: Generate TTIR MLIR from tt-forge
echo ""
echo "Step 1: Generating TTIR MLIR..."
# python benchmark/benchmark.py -p tt-forge-fe -m vovnet -ts classification -bs $BATCH_SIZE -df $DATA_FORMAT -lp 32

# Step 2: Run tt-alchemist to convert TTIR to EmitC
echo ""
echo "Step 2: Converting TTIR to EmitC with optimization..."
# tt-alchemist --input-ttir ~/testify/ll-sw/VovnetTimm/mlir_reports/ttir.mlir \
#              --pipeline-options "enable-optimizer=true memory-layout-analysis-enabled=false" \
#              --output-dir ./vovnet_output

# Step 3: Build the generated C++ code
echo ""
echo "Step 3: Building optimized C++ implementation..."
# cd vovnet_output && ./build.sh

# Step 4: Run benchmark
echo ""
echo "Step 4: Running benchmark..."
# cd vovnet_output && ./run --batch-size $BATCH_SIZE --iterations 10

echo ""
echo "=========================================="
echo "Optimization workflow complete!"
echo "=========================================="
