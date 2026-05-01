#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Run LLMBox Optimizer Models Tests"

llvm-lit -v --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/Silicon/TTNN/llmbox/optimizer
