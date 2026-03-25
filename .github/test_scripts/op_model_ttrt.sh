#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Run Optimizer Models Perf Tests"

# Enable perf feature for lit if running in perf mode
LIT_PARAMS=""
if [ "$1" = "perf" ]; then
    LIT_PARAMS="-D TTMLIR_ENABLE_OPTIMIZER_MODELS_PERF_TESTS=1"
fi

llvm-lit -v $LIT_PARAMS --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
echo
if [ "$IMAGE_NAME" = "speedy" ]; then
    echo "Running optimizer tests"
    $BUILD_DIR/test/unittests/Optimizer/OptimizerTests --gtest_brief=1
    echo
fi
echo "Running op-model ttrt test"
if [ "$1" == "perf" ]; then
    export METAL_HOME=$(ttrt --metal-home-path)
    python -m tracy -r -v --output-folder prof -m ttrt run $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
else
    ttrt $1 $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
fi
cp ${1}_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_optimizer_${TEST_REPORT_PATH##*_} || true
