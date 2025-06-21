#!/bin/bash

# Comprehensive embedding test script for tt-mlir
# Tests embedding-to-linalg conversion and validates functionality

set -e

# Source debug environment
source ./debug_env.sh

echo "=== TT-MLIR Embedding Conversion Test Suite ==="
echo "Testing comprehensive embedding patterns and edge cases"

# Function to run a single test
run_embedding_test() {
    local test_file=$1
    local test_name=$2
    local conversion_pass=$3
    
    echo "----------------------------------------"
    echo "Running test: $test_name"
    echo "File: $test_file"
    echo "Pass: $conversion_pass"
    
    cd build-minimal
    
    # Test the conversion
    if ./bin/ttmlir-opt --$conversion_pass ../$test_file > /tmp/embedding_test_output.mlir 2>&1; then
        echo "✓ $test_name: Conversion succeeded"
        echo "Output written to /tmp/embedding_test_output.mlir"
        
        # Show first few lines of output
        echo "--- Output Preview ---"
        head -20 /tmp/embedding_test_output.mlir
        echo "--- End Preview ---"
    else
        echo "⚠ $test_name: Conversion failed (this may be expected)"
        echo "Error output:"
        cat /tmp/embedding_test_output.mlir
    fi
    
    cd ..
    echo ""
}

# Test 1: Basic embedding decomposition
echo "Creating basic embedding test..."
cat > debug_embedding_basic.mlir << 'EOF'
func.func @test_embedding_basic(%indices: tensor<4xi32>, %weights: tensor<100x128xf32>) -> tensor<4x128xf32> {
  %0 = tensor.empty() : tensor<4x128xf32>
  %result = ttir.embedding %indices, %weights, %0 : tensor<4xi32>, tensor<100x128xf32>, tensor<4x128xf32> -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
EOF

# Test 2: Gather to embedding conversion
echo "Creating gather test that should decompose to embedding..."
cat > debug_gather_to_embedding.mlir << 'EOF'
func.func @test_gather_to_embedding(%input: tensor<100x128xf32>, %indices: tensor<4x1xi32>) -> tensor<4x128xf32> {
  %0 = tensor.empty() : tensor<4x128xf32>
  %result = ttir.gather %input, %indices, %0 : tensor<100x128xf32>, tensor<4x1xi32>, tensor<4x128xf32> -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
EOF

# Test 3: Complex embedding with variable dimensions
echo "Creating complex embedding test..."
cat > debug_embedding_complex.mlir << 'EOF'
func.func @test_embedding_complex(%indices: tensor<8xi32>, %weights: tensor<1000x256xf32>) -> tensor<8x256xf32> {
  %0 = tensor.empty() : tensor<8x256xf32>
  %result = ttir.embedding %indices, %weights, %0 : tensor<8xi32>, tensor<1000x256xf32>, tensor<8x256xf32> -> tensor<8x256xf32>
  return %result : tensor<8x256xf32>
}
EOF

# Run all tests
echo "Running comprehensive embedding conversion tests..."

run_embedding_test "debug_embedding_basic.mlir" "Basic Embedding" "convert-ttir-to-linalg"
run_embedding_test "debug_gather_to_embedding.mlir" "Gather to Embedding" "ttir-to-ttir-decomposition"
run_embedding_test "debug_embedding_complex.mlir" "Complex Embedding" "convert-ttir-to-linalg"

# Test the advanced embedding file we created
if [ -f "test_embedding_advanced.mlir" ]; then
    run_embedding_test "test_embedding_advanced.mlir" "Advanced Embedding Suite" "convert-ttir-to-linalg"
fi

# Test decomposition followed by linalg conversion
echo "----------------------------------------"
echo "Testing full pipeline: gather -> decomposition -> linalg"

cd build-minimal

echo "Step 1: Decompose gather to embedding..."
if ./bin/ttmlir-opt --ttir-to-ttir-decomposition ../debug_gather_to_embedding.mlir > /tmp/decomposed.mlir 2>&1; then
    echo "✓ Decomposition succeeded"
    echo "--- Decomposed Output ---"
    head -15 /tmp/decomposed.mlir
    echo "--- End Decomposed Output ---"
    
    echo "Step 2: Convert decomposed result to linalg..."
    if ./bin/ttmlir-opt --convert-ttir-to-linalg /tmp/decomposed.mlir > /tmp/final_linalg.mlir 2>&1; then
        echo "✓ Full pipeline succeeded"
        echo "--- Final Linalg Output ---"
        head -15 /tmp/final_linalg.mlir
        echo "--- End Final Output ---"
    else
        echo "⚠ Linalg conversion step failed"
        echo "Error:"
        head -20 /tmp/final_linalg.mlir
    fi
else
    echo "⚠ Decomposition step failed"
    echo "Error:"
    head -20 /tmp/decomposed.mlir
fi

cd ..

echo "========================================="
echo "Embedding Test Suite Complete"
echo "Check /tmp/embedding_test_output.mlir, /tmp/decomposed.mlir, and /tmp/final_linalg.mlir for detailed results"
echo "========================================="
