#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# path: path inside test/ttmlir directory to the input ttmlir files
# args: additional arguments to pass to ttrt
# flags: either "run" or "perf"

eval ttrt $3 $2 $BUILD_DIR/test/ttmlir/$1
cp ${3}_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml $TEST_REPORT_PATH || true
