#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi

# Hacky CI fix: EmitC TTNN tests build `libttnn-dylib.so` and link against `_ttnncpp.so`.
# In CI, tt-metal is provided via the tt-mlir install tree, so the shared libs are
# under `$INSTALL_DIR/lib` (not `$TT_METAL_RUNTIME_ROOT/build_Debug/lib`).
if [ -z "${TT_METAL_LIB:-}" ]; then
    export TT_METAL_LIB="$INSTALL_DIR/lib"
fi

export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/perf,$BUILD_DIR/test/python/golden/ttnn
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$BUILD_DIR/test/python/golden/ttnn
pytest -ssv "$@" --junit-xml=$TEST_REPORT_PATH
