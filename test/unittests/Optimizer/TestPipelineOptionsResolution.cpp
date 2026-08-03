// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "llvm/Support/CommandLine.h"

#include "gtest/gtest.h"

#include <optional>

using namespace mlir::tt::ttnn;

// resolveOptimizationLevelOptions() must distinguish "the caller explicitly
// asked for this value" from "nobody touched it, it is still the cl::init
// default". Frontends such as tt-xla assign the options programmatically
// (options.computeCfgMathFidelity = ...), which does NOT bump
// cl::Option::NumOccurrences -- only command-line parsing does. Detecting
// explicit settings via getNumOccurrences() therefore silently discarded every
// programmatically supplied value; hasValue() covers both paths.
//
// These tests exercise the programmatic path specifically, since the lit tests
// only ever reach the options through the command line.
namespace {

// EXPECT_EQ on the Option wrapper itself compares the wrapper, not the value;
// pull the value out explicitly.
OptionalMathFidelity mf(const TTIRToTTNNCommonPipelineOptions &o) {
  return o.computeCfgMathFidelity.getValue();
}
std::optional<bool> fp32(const TTIRToTTNNCommonPipelineOptions &o) {
  return o.computeCfgFp32DestAccEn.getValue();
}

class PipelineOptionsResolutionTest : public ::testing::Test {
protected:
  TTIRToTTNNCommonPipelineOptions options;
};

// Nothing set: the cl::init defaults (HiFi4 / true) survive at optimization
// level 0, so ops exposing a compute kernel config get the accuracy-oriented
// defaults.
TEST_F(PipelineOptionsResolutionTest, DefaultsSurviveAtOptLevelZero) {
  options.optimizationLevel = 0;
  options.resolveOptimizationLevelOptions();

  EXPECT_EQ(mf(options), OptionalMathFidelity::HiFi4);
  EXPECT_EQ(fp32(options), std::optional<bool>(true));
}

// Nothing set: at optimization level > 0 the defaults are cleared so TTNN picks
// per-op defaults. This is the pre-existing behaviour and must not regress.
TEST_F(PipelineOptionsResolutionTest, DefaultsClearedAboveOptLevelZero) {
  options.optimizationLevel = 1;
  options.resolveOptimizationLevelOptions();

  EXPECT_EQ(mf(options), OptionalMathFidelity::Undefined);
  EXPECT_EQ(fp32(options), std::optional<bool>(std::nullopt));
}

// The regression this fixes: a programmatically assigned value must survive at
// optimization level > 0.
TEST_F(PipelineOptionsResolutionTest, ProgrammaticComputeCfgSurvivesOptLevel) {
  options.optimizationLevel = 1;
  options.computeCfgMathFidelity = OptionalMathFidelity::HiFi4;
  options.computeCfgFp32DestAccEn = std::optional<bool>(true);

  // Precondition: assignment must not look like a command-line occurrence,
  // otherwise this test would pass for the wrong reason.
  ASSERT_EQ(options.computeCfgMathFidelity.getNumOccurrences(), 0u);
  ASSERT_EQ(options.computeCfgFp32DestAccEn.getNumOccurrences(), 0u);

  options.resolveOptimizationLevelOptions();

  EXPECT_EQ(mf(options), OptionalMathFidelity::HiFi4);
  EXPECT_EQ(fp32(options), std::optional<bool>(true));
}

// A frontend must be able to force a knob OFF, distinct from leaving it unset.
TEST_F(PipelineOptionsResolutionTest, ProgrammaticFalseIsNotTreatedAsUnset) {
  options.optimizationLevel = 2;
  options.computeCfgFp32DestAccEn = std::optional<bool>(false);
  options.resolveOptimizationLevelOptions();

  EXPECT_EQ(fp32(options), std::optional<bool>(false));
}

// "ttnn_default" maps onto Undefined, which is still an explicit request to not
// override; it must survive rather than be re-derived from the opt level.
TEST_F(PipelineOptionsResolutionTest, ProgrammaticUndefinedSurvives) {
  options.optimizationLevel = 1;
  options.computeCfgMathFidelity = OptionalMathFidelity::Undefined;
  options.resolveOptimizationLevelOptions();

  EXPECT_EQ(mf(options), OptionalMathFidelity::Undefined);
}

// The same predicate guards the non-compute-config options. tt-xla also sets
// enableFusingConv2dWithMultiplyPattern, and that one had no
// `optimizationLevel > 0` guard at all, so it was overridden at every level.
TEST_F(PipelineOptionsResolutionTest, ProgrammaticFusingFlagSurvives) {
  options.optimizationLevel = 1;
  options.enableFusingConv2dWithMultiplyPattern = false;
  options.resolveOptimizationLevelOptions();

  EXPECT_FALSE(options.enableFusingConv2dWithMultiplyPattern);
}

// ... while an untouched flag is still derived from the optimization level.
TEST_F(PipelineOptionsResolutionTest, UntouchedFlagsFollowOptLevel) {
  options.optimizationLevel = 2;
  options.resolveOptimizationLevelOptions();

  EXPECT_TRUE(options.optimizerPassEnabled);
  EXPECT_TRUE(options.enableFusingConv2dWithMultiplyPattern);
  EXPECT_TRUE(options.memoryLayoutAnalysisEnabled);
}

TEST_F(PipelineOptionsResolutionTest,
       ProgrammaticOptimizerPassEnabledSurvives) {
  options.optimizationLevel = 2;
  options.optimizerPassEnabled = false;
  options.resolveOptimizationLevelOptions();

  EXPECT_FALSE(options.optimizerPassEnabled);
}

} // namespace
