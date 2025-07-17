#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to simulate CI regression test job locally
# Based on the "Run Regression Tests" step in .github/workflows/build-and-test.yml

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting regression test simulation...${NC}"

# Find project root using git
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"

# Activate environment
echo -e "${YELLOW}Activating environment...${NC}"
source env/activate

# Set up environment variables
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH}"
export SYSTEM_DESC_PATH="$PROJECT_ROOT/ttrt-artifacts/system_desc.ttsys"
export TTMLIR_ENABLE_REGRESSION_TESTS=1

echo -e "${YELLOW}Environment variables set:${NC}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "SYSTEM_DESC_PATH: $SYSTEM_DESC_PATH"
echo "TTMLIR_ENABLE_REGRESSION_TESTS: $TTMLIR_ENABLE_REGRESSION_TESTS"

# Clean output directory from previous runs
OUTPUT_DIR="$PROJECT_ROOT/build/test/ttmlir/Silicon/TTNN/n150/optimizer/regression_tests/Output"
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Cleaning previous output directory...${NC}"
    rm -rf "$OUTPUT_DIR"/*
fi

# Run llvm-lit on regression tests
echo -e "${YELLOW}Running regression tests with llvm-lit...${NC}"
llvm-lit -v $PROJECT_ROOT/build/test/ttmlir/Silicon/TTNN/n150/optimizer/regression_tests/

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ llvm-lit tests passed${NC}"
else
    echo -e "${RED}✗ llvm-lit tests failed${NC}"
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

echo -e "${GREEN}🎉 All regression tests completed successfully!${NC}"
