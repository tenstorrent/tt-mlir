// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm-gtest/gtest/gtest.h"

#include "gtest/gtest.h"
#include <vector>

using namespace mlir::tt::ttnn;

class Conv2dConfigGeneratorTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  std::function<bool(const Conv2dConfigAttr &)> filterOutFn;
  void SetUp() override {
    context.loadDialect<TTNNDialect>();
    filterOutFn = [](const Conv2dConfigAttr &config) { return false; };
  }
};

TEST_F(Conv2dConfigGeneratorTest, ConstructionMinimal) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  Conv2dConfigSearchSpace space;
  Conv2dConfigGenerator gen(/*op=*/nullptr, baseConfig, space, filterOutFn);
  EXPECT_TRUE(gen.searchDone());
}

TEST_F(Conv2dConfigGeneratorTest, SingleFieldIteration) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  Conv2dConfigSearchSpace space;
  space.weightsDtype = {mlir::tt::ttcore::DataType::BFloat16,
                        mlir::tt::ttcore::DataType::Float32,
                        mlir::tt::ttcore::DataType::UInt32};
  Conv2dConfigGenerator gen(/*op=*/nullptr, baseConfig, space, filterOutFn);
  std::vector<mlir::tt::ttcore::DataType> seen;
  while (!gen.searchDone()) {
    auto config = gen.getNextConfig();
    ASSERT_TRUE(config);
    ASSERT_TRUE(config.getWeightsDtype().has_value());
    seen.push_back(config.getWeightsDtype().value());
  }
  std::vector<mlir::tt::ttcore::DataType> expected = {
      mlir::tt::ttcore::DataType::BFloat16, mlir::tt::ttcore::DataType::Float32,
      mlir::tt::ttcore::DataType::UInt32};
  EXPECT_EQ(seen, expected);
}

TEST_F(Conv2dConfigGeneratorTest, MultipleFieldIteration) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  Conv2dConfigSearchSpace space;
  space.weightsDtype = {mlir::tt::ttcore::DataType::BFloat16,
                        mlir::tt::ttcore::DataType::Float32};
  space.activation = {UnaryOpType::Relu, UnaryOpType::Gelu};
  Conv2dConfigGenerator gen(/*op=*/nullptr, baseConfig, space, filterOutFn);
  std::set<std::pair<mlir::tt::ttcore::DataType, UnaryOpType>> seen;
  while (!gen.searchDone()) {
    auto config = gen.getNextConfig();
    ASSERT_TRUE(config);
    ASSERT_TRUE(config.hasWeightsDtype());
    ASSERT_TRUE(config.hasActivation());
    seen.insert(
        {config.getWeightsDtype().value(), config.getActivation().getOpType()});
  }
  EXPECT_EQ(seen.size(), 4u);
  EXPECT_TRUE(
      seen.count({mlir::tt::ttcore::DataType::BFloat16, UnaryOpType::Relu}));
  EXPECT_TRUE(
      seen.count({mlir::tt::ttcore::DataType::BFloat16, UnaryOpType::Gelu}));
  EXPECT_TRUE(
      seen.count({mlir::tt::ttcore::DataType::Float32, UnaryOpType::Relu}));
  EXPECT_TRUE(
      seen.count({mlir::tt::ttcore::DataType::Float32, UnaryOpType::Gelu}));
}

TEST_F(Conv2dConfigGeneratorTest, FilterOut) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  Conv2dConfigSearchSpace space;
  space.weightsDtype = {mlir::tt::ttcore::DataType::BFloat16,
                        mlir::tt::ttcore::DataType::UInt32};
  space.reshardIfNotOptimal = {true, false};

  // Filter everything out.
  Conv2dConfigGenerator gen(
      /*op=*/nullptr, baseConfig, space,
      [](const Conv2dConfigAttr &config) { return true; });
  auto config = gen.getNextConfig();
  ASSERT_TRUE(!config);
  EXPECT_TRUE(gen.searchDone());
}

TEST_F(Conv2dConfigGeneratorTest, EdgeCaseEmptySearchSpace) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  Conv2dConfigSearchSpace space;
  Conv2dConfigGenerator gen(nullptr, baseConfig, space, filterOutFn);
  EXPECT_TRUE(gen.searchDone());
  auto config = gen.getNextConfig();
  EXPECT_FALSE(config);
}

TEST_F(Conv2dConfigGeneratorTest, EdgeCaseSingleConfig) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  Conv2dConfigSearchSpace space;
  space.weightsDtype = {mlir::tt::ttcore::DataType::Float32};
  Conv2dConfigGenerator gen(nullptr, baseConfig, space, filterOutFn);
  auto config = gen.getNextConfig();
  ASSERT_TRUE(config);
  EXPECT_EQ(config.getWeightsDtype(), mlir::tt::ttcore::DataType::Float32);
  EXPECT_TRUE(gen.searchDone());
  auto config2 = gen.getNextConfig();
  EXPECT_FALSE(config2);
}

TEST_F(Conv2dConfigGeneratorTest, NonEmptyBaseConfig) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
  baseConfig = baseConfig.withWeightsDtype(mlir::tt::ttcore::DataType::Float32);

  Conv2dConfigSearchSpace space;
  space.activation = {UnaryOpType::Relu, UnaryOpType::Gelu};
  space.weightsDtype = {mlir::tt::ttcore::DataType::Float32,
                        mlir::tt::ttcore::DataType::UInt32};
  space.reshardIfNotOptimal = {true, false};

  Conv2dConfigGenerator gen(nullptr, baseConfig, space, filterOutFn);
  std::set<std::tuple<UnaryOpType, mlir::tt::ttcore::DataType, bool>> seen;
  while (!gen.searchDone()) {
    auto config = gen.getNextConfig();
    ASSERT_TRUE(config);
    ASSERT_TRUE(config.hasActivation());
    ASSERT_TRUE(config.hasWeightsDtype());
    ASSERT_TRUE(config.hasReshardIfNotOptimal());
    seen.insert({config.getActivation().getOpType(),
                 config.getWeightsDtype().value(),
                 config.getReshardIfNotOptimal().getValue()});
  }

  EXPECT_EQ(seen.size(), 4u);
  EXPECT_TRUE(seen.count(
      {UnaryOpType::Relu, mlir::tt::ttcore::DataType::Float32, true}));
  EXPECT_TRUE(seen.count(
      {UnaryOpType::Relu, mlir::tt::ttcore::DataType::Float32, false}));
  EXPECT_TRUE(seen.count(
      {UnaryOpType::Gelu, mlir::tt::ttcore::DataType::Float32, true}));
  EXPECT_TRUE(seen.count(
      {UnaryOpType::Gelu, mlir::tt::ttcore::DataType::Float32, false}));
}
