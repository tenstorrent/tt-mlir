#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

# Pick a binary directory that matches the runner. The chisel test suite has
# both binary-driven tests (parametrized over .ttnn files) and builder-driven
# integration tests; the latter use a `device` / `multichip_device` fixture
# that opens the right mesh shape for the current board.
RUNS_ON="${RUNS_ON:-n150}"
BINARY_DIR="$BUILD_DIR/test/ttmlir/Silicon/TTNN/${RUNS_ON}"

pytest test/python/chisel/ \
    --binary "$BINARY_DIR" \
    --junit-xml="$TEST_REPORT_PATH"
