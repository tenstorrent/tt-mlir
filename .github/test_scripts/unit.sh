#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Running unit tests"
llvm-lit -v --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test
echo "Running optimizer tests"
$BUILD_DIR/test/unittests/Optimizer/OptimizerTests --gtest_brief=1
