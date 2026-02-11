#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg1: either "run" or "perf"
# arg2: path inside test/ttmlir directory to the input ttmlir files
# arg3...: additional arguments to pass to ttrt

set -e -o pipefail

echo "Running TTRT tests"
# Resolve the test path, expanding $RUNS_ON then mapping runner names back
# to test directory names (e.g. n300-llmbox runner uses the llmbox test directory)
TTRT_TEST_PATH=$(eval echo "$2")
TTRT_TEST_PATH="${TTRT_TEST_PATH//n300-llmbox/llmbox}"
ttrt "$1" "$BUILD_DIR/test/ttmlir/$TTRT_TEST_PATH" $3
cp ${1}_results.json ${TTRT_REPORT_PATH} || true
cp ttrt_report.xml $TEST_REPORT_PATH || true
