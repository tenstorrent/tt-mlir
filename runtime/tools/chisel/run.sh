#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No folder specified."
    exit 1
fi

# If the arg is a file, use the directory it's in as the arg
if [ -f "$1" ]; then
    INPUT_DIR="$(dirname "$1")"
else
    INPUT_DIR="$1"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 chisel.py \
    --input_dir "$1/" \
    --op_config "$1/op_config.json" \
    --output_dir "$1/output"
