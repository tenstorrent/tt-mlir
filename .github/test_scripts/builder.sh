#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg $1: path to pytest test files
# arg $2: pytest marker expression to select tests to run
# arg $3: "run-ttrt" or predefined additional flags for pytest

set -e -o pipefail

runttrt=""
PYTEST_ARGS=""
FLATBUFFER=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"

for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

# Hacky CI fix: EmitC TTNN tests build `libttnn-dylib.so` and link against `_ttnncpp.so`.
# In CI, tt-metal is provided via the tt-mlir install tree, so the shared libs are
# under `$INSTALL_DIR/lib` (not `$TT_METAL_RUNTIME_ROOT/build_Debug/lib`).
if [ -z "${TT_METAL_LIB:-}" ]; then
    export TT_METAL_LIB="$INSTALL_DIR/lib"
fi

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=${TEST_REPORT_PATH%_*}_builder_${TEST_REPORT_PATH##*_}
