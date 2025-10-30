#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Run Optimizer Models Perf Tests"
llvm-lit -v --filter="optimizer" --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
echo
echo "Running op-model ttrt test"
ttrt perf --non-zero $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
cp perf_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_optimizer_${TEST_REPORT_PATH##*_} || true
