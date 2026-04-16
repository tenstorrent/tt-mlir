#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

pytest -v tools/chisel/tests/ \
    --binary "$BUILD_DIR/test/ttmlir/Silicon/TTNN/n150" \
    --junit-xml="$TEST_REPORT_PATH"
