// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "gtest/gtest.h"

#include <optional>

using namespace mlir::tt::ttnn;

// Options explicitly set by the user must be preserved.
TEST(TTNNPipelineOptionResolution, OptionAlreadySetNoOverrideDirectAssignment) {
  TTIRToTTNNCommonPipelineOptions options;

  options.optimizationLevel = 2;
  options.optimizerPassEnabled = false;
  options.enableFusingConv2dWithMultiplyPattern = false;
  options.memoryLayoutAnalysisEnabled = false;
  options.computeCfgMathFidelity = OptionalMathFidelity::HiFi2;
  options.computeCfgFp32DestAccEn = std::optional<bool>(false);
  options.enableCreateD2MSubgraphs = true;
  options.enableD2MElementwiseFusion = false;

  options.resolveOptimizationLevelOptions();
  options.resolveCreateD2MSubgraphsOptions();

  EXPECT_FALSE(options.optimizerPassEnabled);
  EXPECT_FALSE(options.enableFusingConv2dWithMultiplyPattern);
  EXPECT_FALSE(options.memoryLayoutAnalysisEnabled);
  EXPECT_EQ(options.computeCfgMathFidelity.getValue(),
            OptionalMathFidelity::HiFi2);
  EXPECT_EQ(options.computeCfgFp32DestAccEn.getValue(),
            std::optional<bool>(false));
  EXPECT_FALSE(options.enableD2MElementwiseFusion);
}

// Options explicitly set by the user must be preserved.
TEST(TTNNPipelineOptionResolution, OptionAlreadySetNoOverrideCliParse) {
  TTIRToTTNNCommonPipelineOptions options;

  ASSERT_TRUE(mlir::succeeded(options.parseFromString(
      "optimization-level=2 enable-optimizer=false "
      "enable-fusing-conv2d-with-multiply-pattern=false "
      "memory-layout-analysis-enabled=false "
      "compute-cfg-math-fidelity=hifi2 compute-cfg-fp32-dest-acc-en=false "
      "enable-create-d2m-subgraphs=true "
      "enable-d2m-elementwise-fusion=false")));

  options.resolveOptimizationLevelOptions();
  options.resolveCreateD2MSubgraphsOptions();

  EXPECT_FALSE(options.optimizerPassEnabled);
  EXPECT_FALSE(options.enableFusingConv2dWithMultiplyPattern);
  EXPECT_FALSE(options.memoryLayoutAnalysisEnabled);
  EXPECT_EQ(options.computeCfgMathFidelity.getValue(),
            OptionalMathFidelity::HiFi2);
  EXPECT_EQ(options.computeCfgFp32DestAccEn.getValue(),
            std::optional<bool>(false));
  EXPECT_FALSE(options.enableD2MElementwiseFusion);
}
