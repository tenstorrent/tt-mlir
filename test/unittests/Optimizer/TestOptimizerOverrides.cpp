// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "llvm/Support/CommandLine.h"
#include <gtest/gtest.h>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

using namespace mlir::tt::ttnn;

class OutputLayoutOverrideTest : public ::testing::Test {
protected:
  llvm::cl::opt<std::string> OverrideOutputLayoutOption{
      "override-output-layout"};
  OutputLayoutOverrideParser parser{OverrideOutputLayoutOption};
  llvm::StringMap<OutputLayoutOverrideParams> parsedOverride;
};

TEST_F(OutputLayoutOverrideTest, ParseFullOutputLayoutOverride) {
  std::string arg = "op1=2x2:dram:interleaved:tile:f32";

  bool result = parser.parse(OverrideOutputLayoutOption,
                             "override-output-layout", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 1);
  ASSERT_TRUE(parsedOverride.count("op1"));

  const auto &params = parsedOverride["op1"];
  ASSERT_TRUE(params.grid.has_value());
  ASSERT_EQ(params.grid->size(), 2);
  ASSERT_EQ((*params.grid)[0], 2);
  ASSERT_EQ((*params.grid)[1], 2);
  ASSERT_TRUE(params.bufferType.has_value());
  ASSERT_EQ(params.bufferType.value(), BufferType::DRAM);
  ASSERT_TRUE(params.tensorMemoryLayout.has_value());
  ASSERT_EQ(params.tensorMemoryLayout.value(), TensorMemoryLayout::Interleaved);
  ASSERT_TRUE(params.memoryLayout.has_value());
  ASSERT_EQ(params.memoryLayout.value(), Layout::Tile);
  ASSERT_TRUE(params.dataType.has_value());
  ASSERT_EQ(params.dataType.value(), mlir::tt::DataType::Float32);
}

TEST_F(OutputLayoutOverrideTest, ParsePartialOutputLayoutOverride) {
  std::string arg = "op1=2x2:block_sharded";

  bool result = parser.parse(OverrideOutputLayoutOption,
                             "override-output-layout", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 1);
  ASSERT_TRUE(parsedOverride.count("op1"));

  const auto &params = parsedOverride["op1"];
  ASSERT_TRUE(params.grid.has_value());
  ASSERT_EQ(params.grid->size(), 2);
  ASSERT_EQ((*params.grid)[0], 2);
  ASSERT_EQ((*params.grid)[1], 2);
  ASSERT_FALSE(params.bufferType.has_value());
  ASSERT_TRUE(params.tensorMemoryLayout.has_value());
  ASSERT_EQ(params.tensorMemoryLayout.value(),
            TensorMemoryLayout::BlockSharded);
  ASSERT_FALSE(params.memoryLayout.has_value());
  ASSERT_FALSE(params.dataType.has_value());
}

TEST_F(OutputLayoutOverrideTest, ParseInvalidOutputLayoutOverride) {
  std::string arg = "op1=invalid_value";

  bool result = parser.parse(OverrideOutputLayoutOption,
                             "override-output-layout", arg, parsedOverride);
  ASSERT_TRUE(result);
}

TEST_F(OutputLayoutOverrideTest, ParseMultipleInstancesOfSameParameter) {
  std::string arg = "op1=2x2:2x2";

  bool result = parser.parse(OverrideOutputLayoutOption,
                             "override-output-layout", arg, parsedOverride);
  ASSERT_TRUE(result);
}

TEST_F(OutputLayoutOverrideTest, ParseMultipleOps) {
  std::string arg = "op1=2x2:dram:interleaved:tile:f32,op2=4x4:l1:block_"
                    "sharded:row_major:f16";

  bool result = parser.parse(OverrideOutputLayoutOption,
                             "override-output-layout", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 2);
  ASSERT_TRUE(parsedOverride.count("op1"));
  ASSERT_TRUE(parsedOverride.count("op2"));

  const auto &params1 = parsedOverride["op1"];
  ASSERT_TRUE(params1.grid.has_value());
  ASSERT_EQ(params1.grid->size(), 2);
  ASSERT_EQ((*params1.grid)[0], 2);
  ASSERT_EQ((*params1.grid)[1], 2);
  ASSERT_TRUE(params1.bufferType.has_value());
  ASSERT_EQ(params1.bufferType.value(), BufferType::DRAM);
  ASSERT_TRUE(params1.tensorMemoryLayout.has_value());
  ASSERT_EQ(params1.tensorMemoryLayout.value(),
            TensorMemoryLayout::Interleaved);
  ASSERT_TRUE(params1.memoryLayout.has_value());
  ASSERT_EQ(params1.memoryLayout.value(), Layout::Tile);
  ASSERT_TRUE(params1.dataType.has_value());
  ASSERT_EQ(params1.dataType.value(), mlir::tt::DataType::Float32);

  const auto &params2 = parsedOverride["op2"];
  ASSERT_TRUE(params2.grid.has_value());
  ASSERT_EQ(params2.grid->size(), 2);
  ASSERT_EQ((*params2.grid)[0], 4);
  ASSERT_EQ((*params2.grid)[1], 4);
  ASSERT_TRUE(params2.bufferType.has_value());
  ASSERT_EQ(params2.bufferType.value(), BufferType::L1);
  ASSERT_TRUE(params2.tensorMemoryLayout.has_value());
  ASSERT_EQ(params2.tensorMemoryLayout.value(),
            TensorMemoryLayout::BlockSharded);
  ASSERT_TRUE(params2.memoryLayout.has_value());
  ASSERT_EQ(params2.memoryLayout.value(), Layout::RowMajor);
  ASSERT_TRUE(params2.dataType.has_value());
  ASSERT_EQ(params2.dataType.value(), mlir::tt::DataType::Float16);
}
