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
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  Conv2dConfigSearchSpace space;
  Conv2dConfigGenerator gen(/*op=*/nullptr, baseConfig, space, filterOutFn);
  EXPECT_TRUE(gen.searchDone());
}

TEST_F(Conv2dConfigGeneratorTest, SingleFieldIteration) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  Conv2dConfigSearchSpace space;
  space.dtype = {mlir::tt::ttcore::DataType::BFloat16,
                 mlir::tt::ttcore::DataType::Float32,
                 mlir::tt::ttcore::DataType::UInt32};
  Conv2dConfigGenerator gen(/*op=*/nullptr, baseConfig, space, filterOutFn);
  std::vector<mlir::tt::ttcore::DataType> seen;
  while (!gen.searchDone()) {
    auto config = gen.getNextConfig();
    ASSERT_TRUE(config);
    ASSERT_TRUE(config.getDtype().has_value());
    seen.push_back(config.getDtype().value());
  }
  std::vector<mlir::tt::ttcore::DataType> expected = {
      mlir::tt::ttcore::DataType::BFloat16, mlir::tt::ttcore::DataType::Float32,
      mlir::tt::ttcore::DataType::UInt32};
  EXPECT_EQ(seen, expected);
}

TEST_F(Conv2dConfigGeneratorTest, MultipleFieldIteration) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  Conv2dConfigSearchSpace space;
  space.dtype = {mlir::tt::ttcore::DataType::BFloat16,
                 mlir::tt::ttcore::DataType::Float32};
  space.activation = {"relu", "gelu"};
  Conv2dConfigGenerator gen(/*op=*/nullptr, baseConfig, space, filterOutFn);
  std::set<std::pair<mlir::tt::ttcore::DataType, std::string>> seen;
  while (!gen.searchDone()) {
    auto config = gen.getNextConfig();
    ASSERT_TRUE(config);
    ASSERT_TRUE(config.getDtype().has_value());
    seen.insert({config.getDtype().value(), config.getActivation().str()});
  }
  EXPECT_EQ(seen.size(), 4u);
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::BFloat16, "relu"}));
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::BFloat16, "gelu"}));
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::Float32, "relu"}));
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::Float32, "gelu"}));
}

TEST_F(Conv2dConfigGeneratorTest, FilterOut) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  Conv2dConfigSearchSpace space;
  space.dtype = {mlir::tt::ttcore::DataType::BFloat16,
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
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  Conv2dConfigSearchSpace space;
  Conv2dConfigGenerator gen(nullptr, baseConfig, space, filterOutFn);
  EXPECT_TRUE(gen.searchDone());
  auto config = gen.getNextConfig();
  EXPECT_FALSE(config);
}

TEST_F(Conv2dConfigGeneratorTest, EdgeCaseSingleConfig) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  Conv2dConfigSearchSpace space;
  space.dtype = {mlir::tt::ttcore::DataType::Float32};
  Conv2dConfigGenerator gen(nullptr, baseConfig, space, filterOutFn);
  auto config = gen.getNextConfig();
  ASSERT_TRUE(config);
  EXPECT_EQ(config.getDtype(), mlir::tt::ttcore::DataType::Float32);
  EXPECT_TRUE(gen.searchDone());
  auto config2 = gen.getNextConfig();
  EXPECT_FALSE(config2);
}

TEST_F(Conv2dConfigGeneratorTest, NonEmptyBaseConfig) {
  Conv2dConfigAttr baseConfig = Conv2dConfigAttr::getEmpty(&context);
  baseConfig = baseConfig.withDtype(mlir::tt::ttcore::DataType::Float32);

  Conv2dConfigSearchSpace space;
  space.dtype = {mlir::tt::ttcore::DataType::Float32,
                 mlir::tt::ttcore::DataType::UInt32};
  space.activation = {"relu", "gelu"};
  space.weightsDtype = {mlir::tt::ttcore::DataType::Float32,
                        mlir::tt::ttcore::DataType::UInt32};

  Conv2dConfigGenerator gen(nullptr, baseConfig, space, filterOutFn);
  std::set<std::tuple<mlir::tt::ttcore::DataType, std::string,
                      mlir::tt::ttcore::DataType>>
      seen;
  while (!gen.searchDone()) {
    auto config = gen.getNextConfig();
    ASSERT_TRUE(config);
    seen.insert({config.getDtype().value(), config.getActivation().str(),
                 config.getWeightsDtype().value()});
  }

  EXPECT_EQ(seen.size(), 4u);
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::Float32, "relu",
                          mlir::tt::ttcore::DataType::UInt32}));
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::Float32, "gelu",
                          mlir::tt::ttcore::DataType::UInt32}));
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::Float32, "relu",
                          mlir::tt::ttcore::DataType::Float32}));
  EXPECT_TRUE(seen.count({mlir::tt::ttcore::DataType::Float32, "gelu",
                          mlir::tt::ttcore::DataType::Float32}));
}
