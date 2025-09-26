#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

ttrt $3 $2 $BUILD_DIR/test/ttmlir/$1
if [ "$3" = "perf" ]; then
    # collect ops_perf_results.csv
    cp ttrt_report.xml $PERF_REPORT_PATH
fi
