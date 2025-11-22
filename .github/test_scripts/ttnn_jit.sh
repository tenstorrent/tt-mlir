#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

export PYTHONPATH="$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal"

echo "Running ttnn-jit tests..."
pytest -v $WORK_DIR/test/ttnn-jit/ --junit-xml=$TEST_REPORT_PATH
