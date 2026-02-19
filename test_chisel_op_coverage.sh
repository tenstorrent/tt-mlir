#!/bin/bash
# Quick test suite for chisel op coverage validation
# Tests fundamental operations across different categories

set -e

echo "========================================="
echo "Chisel Op Coverage Test Suite"
echo "========================================="

# Setup environment
source env/activate
export TT_INJECT_TTNN2FB=1
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys

# Create output directory
OUTPUT_DIR="chisel_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo ""
echo "1. Testing Element-wise Unary Operations..."
pytest test/python/golden/ttir_ops/eltwise/test_ttir_unary.py \
    -v -k "exp or sqrt or neg" \
    --save-artifacts \
    --path=$OUTPUT_DIR/eltwise_unary \
    -x || echo "FAILED: Unary ops"

echo ""
echo "2. Testing Element-wise Binary Operations..."
pytest test/python/golden/ttir_ops/eltwise/test_ttir_binary.py \
    -v -k "add or mul or sub" \
    --save-artifacts \
    --path=$OUTPUT_DIR/eltwise_binary \
    -x || echo "FAILED: Binary ops"

echo ""
echo "3. Testing MatMul Operations..."
pytest test/python/golden/ttir_ops/matmul/test_matmul.py \
    -v -k "test_matmul" \
    --save-artifacts \
    --path=$OUTPUT_DIR/matmul \
    -x || echo "FAILED: MatMul"

echo ""
echo "4. Testing Data Movement Operations..."
pytest test/python/golden/ttir_ops/data_movement/test_data_movement.py \
    -v -k "transpose or reshape" \
    --save-artifacts \
    --path=$OUTPUT_DIR/data_movement \
    -x || echo "FAILED: Data movement"

echo ""
echo "5. Testing Reduction Operations..."
pytest test/python/golden/ttir_ops/reduction/test_reduction.py \
    -v -k "sum or mean" \
    --save-artifacts \
    --path=$OUTPUT_DIR/reduction \
    -x || echo "FAILED: Reduction"

echo ""
echo "========================================="
echo "Test Results Summary"
echo "========================================="
echo "Artifacts saved to: $OUTPUT_DIR"
echo ""
echo "To analyze flatbuffers with chisel:"
echo "  python -m chisel <flatbuffer_path>"
echo ""
echo "To test a specific op category again:"
echo "  pytest test/python/golden/ttir_ops/<category>/ -v"
echo "========================================="
