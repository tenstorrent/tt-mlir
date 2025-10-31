#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Run Optimizer Models Perf Tests"
llvm-lit -v --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
echo
if [ "$IMAGE_NAME" = "speedy" ]; then
    echo "Running optimizer tests"
    $BUILD_DIR/test/unittests/Optimizer/OptimizerTests --gtest_brief=1
    echo
fi
echo "Running op-model ttrt test"
ttrt $1 $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
cp ${1}_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_optimizer_${TEST_REPORT_PATH##*_} || true
