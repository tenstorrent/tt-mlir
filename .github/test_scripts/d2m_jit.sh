#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

export PYTHONPATH="$BUILD_DIR/python_packages:$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal:$PYTHONPATH"
mkdir -p $WORK_DIR/third_party/tt-metal
mkdir -p $WORK_DIR/third_party/tt-metal/src
ln -sf $INSTALL_DIR/tt-metal third_party/tt-metal/src/tt-metal

echo "Running d2m-jit tests..."
if [ "$1" == "nightly" ]; then
    # Run perf benchmarks for Superset dashboard
    $WORK_DIR/test/d2m-jit/perf_ci/run_perf_collect.sh "$WORK_DIR/d2m_perf_results" -v --junit-xml=$TEST_REPORT_PATH
else
    pytest -v $WORK_DIR/test/d2m-jit/test_patterns.py --junit-xml=$TEST_REPORT_PATH
fi

# cleanup
rm -rf third_party/tt-metal
