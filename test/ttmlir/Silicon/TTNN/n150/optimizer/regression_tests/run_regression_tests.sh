#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting optimizer models regression tests...${NC}"

# Find project root using git
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"

# Activate environment
echo -e "${YELLOW}Activating environment...${NC}"
source env/activate

# Set up environment variables (CI sets LD_LIBRARY_PATH and SYSTEM_DESC_PATH)
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:${TTMLIR_TOOLCHAIN_DIR}/lib"
fi
if [ -z "$SYSTEM_DESC_PATH" ]; then
    export SYSTEM_DESC_PATH="$PROJECT_ROOT/ttrt-artifacts/system_desc.ttsys"
fi
export TTMLIR_ENABLE_OPTIMIZER_MODELS_REGRESSION_TESTS=1

echo -e "${YELLOW}Environment variables set:${NC}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "SYSTEM_DESC_PATH: $SYSTEM_DESC_PATH"
echo "TTMLIR_ENABLE_OPTIMIZER_MODELS_REGRESSION_TESTS: $TTMLIR_ENABLE_OPTIMIZER_MODELS_REGRESSION_TESTS"

# Check if TTMLIR_ENABLE_OPMODEL was enabled during build
echo -e "${YELLOW}Checking if TTMLIR_ENABLE_OPMODEL is enabled...${NC}"
CMAKE_CACHE_FILE="$PROJECT_ROOT/build/CMakeCache.txt"
if [ ! -f "$CMAKE_CACHE_FILE" ]; then
    echo -e "${RED}✗ ERROR: CMakeCache.txt not found. Please build the project first.${NC}"
    exit 1
fi

if ! grep -q "TTMLIR_ENABLE_OPMODEL:BOOL=ON" "$CMAKE_CACHE_FILE"; then
    echo -e "${RED}✗ ERROR: TTMLIR_ENABLE_OPMODEL is not enabled in build.${NC}"
    echo -e "${RED}Please rebuild with -DTTMLIR_ENABLE_OPMODEL=ON${NC}"
    exit 1
fi
echo -e "${GREEN}✓ TTMLIR_ENABLE_OPMODEL is enabled${NC}"

# Clean output directory from previous runs
OUTPUT_DIR="$PROJECT_ROOT/build/test/ttmlir/Silicon/TTNN/n150/optimizer/regression_tests/Output"
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Cleaning previous output directory...${NC}"
    rm -rf "$OUTPUT_DIR"/*
fi

# Run llvm-lit on regression tests
echo -e "${YELLOW}Running optimizer models regression tests with llvm-lit...${NC}"
llvm-lit -v $PROJECT_ROOT/build/test/ttmlir/Silicon/TTNN/n150/optimizer/regression_tests/

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ llvm-lit optimizer models tests passed${NC}"
else
    echo -e "${RED}✗ llvm-lit optimizer models tests failed${NC}"
    exit 1
fi

# Run ttrt on generated flatbuffers
echo -e "${YELLOW}Running ttrt on generated flatbuffers...${NC}"
cd $PROJECT_ROOT/build/test/ttmlir/Silicon/TTNN/n150/optimizer/regression_tests/Output/

# Count flatbuffer files
ttnn_count=$(ls *.tmp.ttnn 2>/dev/null | wc -l)
if [ $ttnn_count -eq 0 ]; then
    echo -e "${RED}✗ No flatbuffer files found (*.tmp.ttnn)${NC}"
    exit 1
fi

echo -e "${YELLOW}Found $ttnn_count flatbuffer files to execute${NC}"

# Execute each flatbuffer with ttrt (fail fast approach)
for ttnn_file in *.tmp.ttnn; do
    if [ -f "$ttnn_file" ]; then
        echo -e "${YELLOW}Executing $ttnn_file with ttrt...${NC}"
        if ttrt run "$ttnn_file"; then
            echo -e "${GREEN}✓ $ttnn_file executed successfully${NC}"
        else
            echo -e "${RED}✗ ERROR: ttrt execution failed for $ttnn_file${NC}"
            echo -e "${RED}Stopping execution due to failure${NC}"
            exit 1
        fi
        echo "---"
    fi
done

echo -e "${GREEN}🎉 All optimizer models regression tests completed successfully!${NC}"
