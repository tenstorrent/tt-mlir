#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

export PYTHONPATH="$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal"

echo "Running ttnn-jit tests..."
echo "Checking Python path and installations..."
python -c "import sys; print('sys.path:', sys.path)"
echo "Checking for ttmlir packages..."
python -c "import sys; [print(f'ttmlir found in: {p}') for p in sys.path if __import__('os').path.exists(__import__('os').path.join(p, 'ttmlir'))]" || true
echo "Checking ttnn-jit installation..."
python -c "import ttnn_jit; print(f'ttnn-jit imported from: {ttnn_jit.__file__}')"
echo "Starting pytest..."
pytest -sv $WORK_DIR/test/ttnn-jit/ --junit-xml=$TEST_REPORT_PATH
