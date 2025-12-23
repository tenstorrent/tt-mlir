#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg1: either "run" or "perf"
# arg2: path inside test/ttmlir directory to the input ttmlir files
# arg3...: additional arguments to pass to ttrt

set -e -o pipefail

echo "Running TTRT tests"
eval ttrt "$1" "$BUILD_DIR/test/ttmlir/$2" "$3"

# Copy results only if files exist
if [ -f "${1}_results.json" ]; then
	cp "${1}_results.json" "${TTRT_REPORT_PATH}"
fi

if [ -f "ttrt_report.xml" ]; then
	cp "ttrt_report.xml" "$TEST_REPORT_PATH"
fi
