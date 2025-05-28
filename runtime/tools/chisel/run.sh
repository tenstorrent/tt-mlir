#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No folder specified."
    exit 1
fi

python3 chisel.py \
    --input_dir $1/ \
    --op_config $1/op_config.json \
    --output_dir $1/output
