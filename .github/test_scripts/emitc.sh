#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

llvm-lit -sv --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/EmitC/TTNN
ttrt run --emitc $BUILD_DIR/test/ttmlir/EmitC
cp run_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_emitc_${TEST_REPORT_PATH##*_} || true
