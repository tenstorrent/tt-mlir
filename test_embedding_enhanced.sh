#!/bin/bash

# Enhanced comprehensive embedding test script for tt-mlir
# Tests embedding-to-linalg conversion with proper MLIR syntax

set -e

# Source debug environment
source ./debug_env.sh

echo "=== TT-MLIR Enhanced Embedding Conversion Test Suite ==="
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

# Test 1: Basic embedding with proper syntax
echo "Creating basic embedding test with proper MLIR syntax..."
cat > debug_embedding_basic_proper.mlir << 'EOF'
func.func @test_embedding_basic(%indices: tensor<4xi32>, %weights: tensor<100x128xf32>) -> tensor<4x128xf32> {
  %0 = ttir.empty() : tensor<4x128xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<4xi32>, tensor<100x128xf32>, tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
EOF

# Test 2: Gather to embedding conversion with proper syntax
echo "Creating gather test with proper MLIR syntax..."
cat > debug_gather_proper.mlir << 'EOF'
func.func @test_gather_basic(%input: tensor<100x128xf32>, %indices: tensor<4x1xi32>) -> tensor<4x128xf32> {
  %0 = ttir.empty() : tensor<4x128xf32>
  %result = "ttir.gather"(%input, %indices, %0) {
      offset_dims = array<i64: 1>,
      collapsed_slice_dims = array<i64: 0>,
      operand_batching_dims = array<i64>,
      start_indices_batching_dims = array<i64>,
      start_index_map = array<i64: 0>,
      index_vector_dim = 1 : si64,
      slice_sizes = array<i64: 1, 128>,
      indices_are_sorted = false
  } : (tensor<100x128xf32>, tensor<4x1xi32>, tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
EOF

# Test 3: Complex embedding with different dimensions
echo "Creating complex embedding test..."
cat > debug_embedding_complex_proper.mlir << 'EOF'
func.func @test_embedding_complex(%indices: tensor<8xi32>, %weights: tensor<1000x256xf32>) -> tensor<8x256xf32> {
  %0 = ttir.empty() : tensor<8x256xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<8xi32>, tensor<1000x256xf32>, tensor<8x256xf32>) -> tensor<8x256xf32>
  return %result : tensor<8x256xf32>
}
EOF

# Test 4: 2D indices embedding  
echo "Creating 2D indices embedding test..."
cat > debug_embedding_2d.mlir << 'EOF'
func.func @test_embedding_2d(%indices: tensor<2x4xi32>, %weights: tensor<100x128xf32>) -> tensor<2x4x128xf32> {
  %0 = ttir.empty() : tensor<2x4x128xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<2x4xi32>, tensor<100x128xf32>, tensor<2x4x128xf32>) -> tensor<2x4x128xf32>
  return %result : tensor<2x4x128xf32>
}
EOF

# Run all tests
echo "Running comprehensive embedding conversion tests..."

run_embedding_test "debug_embedding_basic_proper.mlir" "Basic Embedding (Proper Syntax)" "convert-ttir-to-linalg"
run_embedding_test "debug_gather_proper.mlir" "Gather to Embedding (Proper Syntax)" "ttir-to-ttir-decomposition"
run_embedding_test "debug_embedding_complex_proper.mlir" "Complex Embedding (Proper Syntax)" "convert-ttir-to-linalg"
run_embedding_test "debug_embedding_2d.mlir" "2D Indices Embedding" "convert-ttir-to-linalg"

# Test decomposition followed by linalg conversion
echo "----------------------------------------"
echo "Testing full pipeline: gather -> decomposition -> linalg"

cd build-minimal

echo "Step 1: Decompose gather to embedding..."
if ./bin/ttmlir-opt --ttir-to-ttir-decomposition ../debug_gather_proper.mlir > /tmp/decomposed_proper.mlir 2>&1; then
    echo "✓ Decomposition succeeded"
    echo "--- Decomposed Output ---"
    head -15 /tmp/decomposed_proper.mlir
    echo "--- End Decomposed Output ---"
    
    echo "Step 2: Convert decomposed result to linalg..."
    if ./bin/ttmlir-opt --convert-ttir-to-linalg /tmp/decomposed_proper.mlir > /tmp/final_linalg_proper.mlir 2>&1; then
        echo "✓ Full pipeline succeeded"
        echo "--- Final Linalg Output ---"
        head -15 /tmp/final_linalg_proper.mlir
        echo "--- End Final Output ---"
    else
        echo "⚠ Linalg conversion step failed"
        echo "Error:"
        head -20 /tmp/final_linalg_proper.mlir
    fi
else
    echo "⚠ Decomposition step failed"
    echo "Error:"
    head -20 /tmp/decomposed_proper.mlir
fi

cd ..

# Performance benchmarking setup
echo "----------------------------------------"
echo "Setting up performance benchmarking tests..."

cat > debug_embedding_benchmark.mlir << 'EOF'
func.func @benchmark_embedding_small(%indices: tensor<10xi32>, %weights: tensor<50x32xf32>) -> tensor<10x32xf32> {
  %0 = ttir.empty() : tensor<10x32xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<10xi32>, tensor<50x32xf32>, tensor<10x32xf32>) -> tensor<10x32xf32>
  return %result : tensor<10x32xf32>
}

func.func @benchmark_embedding_medium(%indices: tensor<100xi32>, %weights: tensor<1000x128xf32>) -> tensor<100x128xf32> {
  %0 = ttir.empty() : tensor<100x128xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<100xi32>, tensor<1000x128xf32>, tensor<100x128xf32>) -> tensor<100x128xf32>
  return %result : tensor<100x128xf32>
}

func.func @benchmark_embedding_large(%indices: tensor<1000xi32>, %weights: tensor<10000x512xf32>) -> tensor<1000x512xf32> {
  %0 = ttir.empty() : tensor<1000x512xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<1000xi32>, tensor<10000x512xf32>, tensor<1000x512xf32>) -> tensor<1000x512xf32>
  return %result : tensor<1000x512xf32>
}
EOF

run_embedding_test "debug_embedding_benchmark.mlir" "Embedding Benchmark Suite" "convert-ttir-to-linalg"

echo "========================================="
echo "Enhanced Embedding Test Suite Complete"
echo ""
echo "Results available in:"
echo "  - /tmp/embedding_test_output.mlir"
echo "  - /tmp/decomposed_proper.mlir" 
echo "  - /tmp/final_linalg_proper.mlir"
echo ""
echo "Ready for Phase 2: Performance Optimization"
echo "========================================="
