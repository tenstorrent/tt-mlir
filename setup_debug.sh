#!/bin/bash

# Debug Environment Setup Script for tt-mlir Issue #3849
# This script sets up a comprehensive debugging environment for the CPU Fallback Gather Op Crash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TT-MLIR Debug Environment Setup${NC}"
echo -e "${BLUE}Issue #3849: CPU Fallback Gather Op Crash${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Activate environment
echo -e "\n${YELLOW}Step 1: Activating tt-mlir environment...${NC}"
source env/activate
echo -e "${GREEN}✓ Environment activated${NC}"

# Step 2: Set debug environment variables
echo -e "\n${YELLOW}Step 2: Setting debug environment variables...${NC}"
export CMAKE_BUILD_TYPE="Debug"
export TTMLIR_ENABLE_DEBUG_LOGS="ON"
export MLIR_ENABLE_DUMP=1
export MLIR_ENABLE_TIMING=1
export MLIR_ENABLE_CRASH_REPRODUCER=1
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=1:check_initialization_order=1"
export UBSAN_OPTIONS="print_stacktrace=1:abort_on_error=1"

echo -e "${GREEN}✓ Debug environment variables set${NC}"
echo "  - CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
echo "  - TTMLIR_ENABLE_DEBUG_LOGS=$TTMLIR_ENABLE_DEBUG_LOGS"
echo "  - MLIR_ENABLE_DUMP=$MLIR_ENABLE_DUMP"
echo "  - MLIR_ENABLE_TIMING=$MLIR_ENABLE_TIMING"

# Step 3: Create debug build directory
echo -e "\n${YELLOW}Step 3: Setting up debug build directory...${NC}"
mkdir -p build-debug
cd build-debug

# Step 4: Configure CMake for debugging
echo -e "\n${YELLOW}Step 4: Configuring CMake for debug build...${NC}"
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTTMLIR_ENABLE_DEBUG_LOGS=ON \
  -DCMAKE_C_FLAGS="-g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer" \
  -DCMAKE_LINKER_FLAGS="-fsanitize=address -fsanitize=undefined" \
  -DTTMLIR_ENABLE_RUNTIME=OFF \
  -DTTMLIR_ENABLE_PYKERNEL=ON \
  -DTTMLIR_ENABLE_STABLEHLO=OFF \
  -DTTMLIR_ENABLE_OPMODEL=OFF \
  -DCODE_COVERAGE=ON

echo -e "${GREEN}✓ CMake configured for debug build${NC}"

# Step 5: Build with debug info
echo -e "\n${YELLOW}Step 5: Building debug version...${NC}"
make -j$(nproc) ttmlir-opt ttmlir-translate

echo -e "${GREEN}✓ Debug build completed${NC}"

# Step 6: Create debugging utilities
echo -e "\n${YELLOW}Step 6: Creating debugging utilities...${NC}"

# Create a script for running gather tests with debugging
cat > debug_gather_test.sh << 'EOF'
#!/bin/bash
# Debug script for gather op testing

source ../env/activate
export MLIR_ENABLE_DUMP=1
export MLIR_ENABLE_TIMING=1

echo "=== Running gather test with debugging ==="
echo "Current directory: $(pwd)"
echo "MLIR tools directory: $(which ttmlir-opt)"

# Test TTIR to TTIR decomposition
echo -e "\n--- Testing TTIR to TTIR Decomposition ---"
echo "Command: ttmlir-opt --ttir-to-ttir-decomposition debug_gather.mlir"
if [ -f "../debug_gather.mlir" ]; then
    ./bin/ttmlir-opt --ttir-to-ttir-decomposition ../debug_gather.mlir 2>&1 | tee decomposition_output.log
else
    echo "Note: debug_gather.mlir not found, will be created"
fi

# Test TTIR to Linalg conversion
echo -e "\n--- Testing TTIR to Linalg Conversion ---"
echo "Command: ttmlir-opt --convert-ttir-to-linalg debug_gather.mlir"
if [ -f "../debug_gather.mlir" ]; then
    ./bin/ttmlir-opt --convert-ttir-to-linalg ../debug_gather.mlir 2>&1 | tee linalg_output.log
else
    echo "Note: debug_gather.mlir not found, will be created"
fi

# Run Python golden tests with debugging
echo -e "\n--- Running Python Golden Tests ---"
cd ../test/python/golden
if [ -f "test_ttir_ops.py" ]; then
    echo "Running: python test_ttir_ops.py -k gather -v --tb=long"
    python test_ttir_ops.py -k gather -v --tb=long 2>&1 | tee ../../../build-debug/gather_test_output.log
else
    echo "Warning: test_ttir_ops.py not found"
fi

cd ../../../build-debug
echo -e "\n=== Debug session completed ==="
echo "Logs saved:"
echo "  - decomposition_output.log"
echo "  - linalg_output.log" 
echo "  - gather_test_output.log"
EOF

chmod +x debug_gather_test.sh

# Create minimal test case
cat > ../debug_gather.mlir << 'EOF'
// Minimal gather test case for debugging Issue #3849
func.func @test_gather_simple(%input: tensor<4x3xf32>, %indices: tensor<2xi32>) -> tensor<2x3xf32> {
  %0 = ttir.empty() : tensor<2x3xf32>
  %1 = "ttir.gather"(%input, %indices, %0) {
    collapsed_slice_dims = array<i64: 0>,
    index_vector_dim = 1 : si64,
    offset_dims = array<i64: 1>,
    slice_sizes = array<i64: 1, 3>,
    start_index_map = array<i64: 0>
  } : (tensor<4x3xf32>, tensor<2xi32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// More complex gather case
func.func @test_gather_complex(%input: tensor<8x4x3xf32>, %indices: tensor<3x2xi32>) -> tensor<3x4x3xf32> {
  %0 = ttir.empty() : tensor<3x4x3xf32>
  %1 = "ttir.gather"(%input, %indices, %0) {
    collapsed_slice_dims = array<i64: 0>,
    index_vector_dim = 2 : si64,
    offset_dims = array<i64: 1, 2>,
    slice_sizes = array<i64: 1, 4, 3>,
    start_index_map = array<i64: 0>
  } : (tensor<8x4x3xf32>, tensor<3x2xi32>, tensor<3x4x3xf32>) -> tensor<3x4x3xf32>
  return %1 : tensor<3x4x3xf32>
}
EOF

# Create GDB debug script
cat > debug_with_gdb.sh << 'EOF'
#!/bin/bash
# GDB debugging script for gather op crash

source ../env/activate

echo "Starting GDB session for gather op debugging..."
echo "Useful GDB commands:"
echo "  - break mlir::LogicalResult"
echo "  - break ttir::GatherOp"
echo "  - run --convert-ttir-to-linalg ../debug_gather.mlir"
echo "  - bt (backtrace)"
echo "  - print *op"

gdb --args ./bin/ttmlir-opt --convert-ttir-to-linalg ../debug_gather.mlir
EOF

chmod +x debug_with_gdb.sh

echo -e "${GREEN}✓ Debugging utilities created${NC}"
echo "  - debug_gather_test.sh: Run gather tests with debug output"
echo "  - debug_with_gdb.sh: Start GDB debugging session"
echo "  - ../debug_gather.mlir: Minimal test cases"

cd ..

# Step 7: Summary and next steps
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Debug Environment Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Quick Start Commands:${NC}"
echo "1. Enter debug build directory:"
echo "   cd build-debug"
echo ""
echo "2. Run debug tests:"
echo "   ./debug_gather_test.sh"
echo ""
echo "3. Start GDB debugging:"
echo "   ./debug_with_gdb.sh"
echo ""
echo "4. Manual MLIR testing:"
echo "   ./bin/ttmlir-opt --help"
echo "   ./bin/ttmlir-opt --convert-ttir-to-linalg ../debug_gather.mlir"
echo ""

echo -e "${YELLOW}Debug Files Created:${NC}"
echo "  - debug_gather.mlir: Test cases for reproduction"
echo "  - build-debug/: Debug build directory with sanitizers"
echo "  - build-debug/debug_gather_test.sh: Automated test runner"
echo "  - build-debug/debug_with_gdb.sh: GDB debugging helper"

echo -e "\n${YELLOW}Next Steps for Issue #3849:${NC}"
echo "1. Run './debug_gather_test.sh' to reproduce the crash"
echo "2. Examine the logs for crash location"
echo "3. Use GDB to get detailed stack trace"
echo "4. Check lib/Conversion/TTIRToLinalg/ for missing gather pattern"
echo "5. Implement the missing GatherOpConversionPattern"

echo -e "\n${GREEN}Happy debugging! 🐛${NC}"
