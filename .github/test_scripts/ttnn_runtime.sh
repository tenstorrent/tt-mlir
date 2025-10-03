#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Generating tests"
llvm-lit $BUILD_DIR/test
echo "Running TTNN Runtime Python tests"
.github/test_scripts/pytest.sh "$1" "$2" "$3"
