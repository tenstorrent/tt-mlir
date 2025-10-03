#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

llvm-lit -sv --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/EmitC/TTNN
ttrt run --emitc $BUILD_DIR/test/ttmlir/$1
cp run_results.json ${TTRT_REPORT_PATH} || true
