#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

llvm-lit -sv --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/EmitPy
ttrt emitpy $BUILD_DIR/test/ttmlir/EmitPy
cp emitpy_results.json ${TTRT_REPORT_PATH} || true
