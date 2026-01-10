#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to update single block and single layer perf tests from llm_blocks_and_layers directory
# Usage: ./update_llm_perf_tests.sh <llm_blocks_and_layers_YYMMDD>
# Example: ./update_llm_perf_tests.sh llm_blocks_and_layers_251223

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Check for source directory argument
if [ -z "$1" ]; then
    echo "Usage: $0 <llm_blocks_and_layers_directory>"
    echo "Example: $0 llm_blocks_and_layers_251223"
    exit 1
fi

SOURCE_DIR="${REPO_ROOT}/$1"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory '$SOURCE_DIR' does not exist"
    exit 1
fi

MODELS_DIR="${REPO_ROOT}/test/ttmlir/models"
LLM_MODELS_DIR="${MODELS_DIR}/llm_blocks_and_layers"
LLM_TESTS_DIR="${REPO_ROOT}/test/ttmlir/Silicon/TTNN/n150/optimizer/llm_block_layer_perf_tests"

# Create directories if they don't exist
mkdir -p "$LLM_MODELS_DIR"
mkdir -p "$LLM_TESTS_DIR"

# Create test file only if it doesn't already exist
# Returns 0 if test existed, 1 if created new
create_test_file_if_missing() {
    local model_name="$1"
    local test_file="${LLM_TESTS_DIR}/${model_name}.mlir"

    if [ -f "$test_file" ]; then
        return 0  # Test existed
    fi

    cat > "$test_file" << EOF
// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o ${model_name}_ttnn.mlir %models/llm_blocks_and_layers/${model_name}.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn ${model_name}_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
EOF
    return 1  # Created new
}

# Create lit.local.cfg files if they don't exist
create_lit_config() {
    local dir="$1"
    local config_file="${dir}/lit.local.cfg"
    if [ ! -f "$config_file" ]; then
        cat > "$config_file" << 'EOF'
# Configuration for opmodel tests in this subdirectory
# These tests require single-threaded execution due to physical device access
# when opmodel is enabled.

config.parallelism_group = "opmodel"
EOF
        echo "  Created: $config_file"
    fi
}

echo "Updating LLM perf tests from: $SOURCE_DIR"
echo ""

# Clean up outdated models and tests (only remove if not in source directory)
removed_count=0
for existing_model in "${LLM_MODELS_DIR}"/*.mlir; do
    if [ -f "$existing_model" ]; then
        model_name=$(basename "$existing_model" .mlir)
        if [ ! -f "${SOURCE_DIR}/${model_name}.mlir" ]; then
            rm -f "$existing_model"
            rm -f "${LLM_TESTS_DIR}/${model_name}.mlir"
            echo "  [REMOVED]  $model_name"
            removed_count=$((removed_count + 1))
        fi
    fi
done

# Create lit.local.cfg file
echo ""
echo "Creating lit.local.cfg..."
create_lit_config "$LLM_TESTS_DIR"

# Process all .mlir files
echo ""
new_count=0
updated_count=0

for model_file in "${SOURCE_DIR}"/*.mlir; do
    if [ -f "$model_file" ]; then
        model_name=$(basename "$model_file" .mlir)

        # Only process block and layer models
        if [[ "$model_name" == *_block ]] || [[ "$model_name" == *_layer ]]; then
            cp "$model_file" "$LLM_MODELS_DIR/"
            if create_test_file_if_missing "$model_name"; then
                echo "  [UPDATED] $model_name"
                updated_count=$((updated_count + 1))
            else
                echo "  [NEW]     $model_name"
                new_count=$((new_count + 1))
            fi
        fi
    fi
done

echo ""
echo "Summary:"
echo "  [NEW]     $new_count"
echo "  [UPDATED] $updated_count"
echo "  [REMOVED] $removed_count"
echo ""
echo "Directories:"
echo "  Models: $LLM_MODELS_DIR"
echo "  Tests:  $LLM_TESTS_DIR"
