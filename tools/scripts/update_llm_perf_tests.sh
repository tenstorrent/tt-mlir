#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to update single block and single layer perf tests from transformer_test_irs directory
# Usage: ./update_llm_perf_tests.sh <source_directory> <model1> [model2] [model3] ...
# Example: ./update_llm_perf_tests.sh transformer_test_irs llama_3_2_1b_decode_block falcon_3_1b_prefill_layer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Check for source directory argument
if [ -z "$1" ]; then
    echo "Usage: $0 <source_directory> <model1> [model2] [model3] ..."
    echo "       $0 <source_directory> --list           # List available models"
    echo "       $0 <source_directory> --models-only    # Copy all models without creating test files"
    echo ""
    echo "Example: $0 transformer_test_irs llama_3_2_1b_decode_block falcon_3_1b_prefill_layer"
    exit 1
fi

SOURCE_DIR="${REPO_ROOT}/$1"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory '$SOURCE_DIR' does not exist"
    exit 1
fi

shift  # Remove source directory from arguments

MODELS_DIR="${REPO_ROOT}/test/ttmlir/models"
LLM_MODELS_DIR="${MODELS_DIR}/single_blocks_and_layers"
LLM_TESTS_DIR="${REPO_ROOT}/test/ttmlir/Silicon/TTNN/n150/optimizer/single_block_layer_perf_tests"

# List available models if --list is passed
if [ "$1" = "--list" ]; then
    echo "Available models in $SOURCE_DIR:"
    echo ""
    for model_file in "${SOURCE_DIR}"/*.mlir; do
        if [ -f "$model_file" ]; then
            model_name=$(basename "$model_file" .mlir)
            if [[ "$model_name" == *_block ]] || [[ "$model_name" == *_layer ]]; then
                # Check if already added
                if [ -f "${LLM_MODELS_DIR}/${model_name}.mlir" ]; then
                    echo "  [EXISTS] $model_name"
                else
                    echo "  [NEW]    $model_name"
                fi
            fi
        fi
    done
    exit 0
fi

# Copy all models without creating test files if --models-only is passed
if [ "$1" = "--models-only" ]; then
    echo "Copying models from: $SOURCE_DIR"
    echo "(Models only - no test files will be created)"
    echo ""

    mkdir -p "$LLM_MODELS_DIR"

    new_count=0
    updated_count=0

    for model_file in "${SOURCE_DIR}"/*.mlir; do
        if [ -f "$model_file" ]; then
            model_name=$(basename "$model_file" .mlir)
            if [[ "$model_name" == *_block ]] || [[ "$model_name" == *_layer ]]; then
                if [ -f "${LLM_MODELS_DIR}/${model_name}.mlir" ]; then
                    echo "  [UPDATED] $model_name"
                    updated_count=$((updated_count + 1))
                else
                    echo "  [NEW]     $model_name"
                    new_count=$((new_count + 1))
                fi
                cp "$model_file" "$LLM_MODELS_DIR/"
            fi
        fi
    done

    echo ""
    echo "Summary:"
    echo "  [NEW]     $new_count"
    echo "  [UPDATED] $updated_count"
    echo ""
    echo "Models copied to: $LLM_MODELS_DIR"
    exit 0
fi

# Check that at least one model is specified
if [ $# -eq 0 ]; then
    echo "Error: No models specified"
    echo ""
    echo "Usage: $0 <source_directory> <model1> [model2] [model3] ..."
    echo "       $0 <source_directory> --list           # List available models"
    echo "       $0 <source_directory> --models-only    # Copy all models without creating test files"
    exit 1
fi

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
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o ${model_name}_ttnn.mlir %models/single_blocks_and_layers/${model_name}.mlir
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

echo "Adding specified models from: $SOURCE_DIR"
echo ""

# Create lit.local.cfg file
create_lit_config "$LLM_TESTS_DIR"

# Process specified models
new_count=0
updated_count=0
error_count=0

for model_name in "$@"; do
    # Remove .mlir extension if provided
    model_name="${model_name%.mlir}"

    model_file="${SOURCE_DIR}/${model_name}.mlir"

    if [ ! -f "$model_file" ]; then
        echo "  [ERROR]   $model_name - not found in source directory"
        error_count=$((error_count + 1))
        continue
    fi

    # Validate it's a block or layer model
    if [[ "$model_name" != *_block ]] && [[ "$model_name" != *_layer ]]; then
        echo "  [SKIP]    $model_name - must end with _block or _layer"
        continue
    fi

    cp "$model_file" "$LLM_MODELS_DIR/"
    if create_test_file_if_missing "$model_name"; then
        echo "  [UPDATED] $model_name"
        updated_count=$((updated_count + 1))
    else
        echo "  [NEW]     $model_name"
        new_count=$((new_count + 1))
    fi
done

echo ""
echo "Summary:"
echo "  [NEW]     $new_count"
echo "  [UPDATED] $updated_count"
echo "  [ERROR]   $error_count"
echo ""
echo "Directories:"
echo "  Models: $LLM_MODELS_DIR"
echo "  Tests:  $LLM_TESTS_DIR"
