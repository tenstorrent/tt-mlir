#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg1: either "run" or "perf"
# arg2: path inside test/ttmlir directory to the input ttmlir files
# arg3...: additional arguments to pass to ttrt

set -e -o pipefail

echo "Generating tests"
llvm-lit --filter="Silicon" $BUILD_DIR/test

echo "Running TTRT tests"
eval ttrt "$1" "$BUILD_DIR/test/ttmlir/$2" "$3"
cp ${1}_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml $TEST_REPORT_PATH || true
