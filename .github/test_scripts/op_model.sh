#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Running optimizer tests"
$BUILD_DIR/test/unittests/Optimizer/OptimizerTests --gtest_brief=1
echo "Running op-model test Conversion"
$BUILD_DIR/test/unittests/OpModel/TTNN/Conversion/TestConversion
echo "Running op-model test Lib"
$BUILD_DIR/test/unittests/OpModel/TTNN/Lib/TestOpModelLib
echo "Running op-model test Interface"
$BUILD_DIR/test/unittests/OpModel/TTNN/Op/TestOpModelInterface
echo
echo "Run Optimizer Models Perf Tests"
llvm-lit -v --xunit-xml-output $TEST_REPORT_PATH --param TTMLIR_ENABLE_OPTIMIZER_MODELS_PERF_TESTS=1 $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer/models_perf_tests
