#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

llvm-lit -sv --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/EmitC/TTNN
ttrt emitc $BUILD_DIR/test/ttmlir/EmitC
cp emitc_results.json ${TTRT_REPORT_PATH} || true
