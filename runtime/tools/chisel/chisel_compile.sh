#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No folder specified."
    exit 1
fi

ttmlir-opt --ttir-to-ttir-decomposition $1/ttir_wide.mlir > $1/ttir.mlir
ttmlir-opt --mlir-print-debuginfo --tt-register-device="system-desc-path=./ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-erase-inverse-ops-pass=false" $1/ttir.mlir > $1/ttnn.mlir
ttmlir-translate --ttnn-to-flatbuffer $1/ttnn.mlir -o $1/fb.ttnn
