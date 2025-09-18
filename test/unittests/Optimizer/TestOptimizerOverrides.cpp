// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

#include "llvm/Support/CommandLine.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;

class Conv2dConfigOverrideTest : public ::testing::Test {
protected:
  static llvm::cl::opt<std::string> OverrideConv2dConfigOption;
  Conv2dConfigOverrideParser parser{OverrideConv2dConfigOption};
  llvm::StringMap<Conv2dConfigOverrideParams> parsedOverride;
};
class OutputLayoutOverrideTest : public ::testing::Test {
protected:
  static llvm::cl::opt<std::string> OverrideOutputLayoutOption;
  OutputLayoutOverrideParser parser{OverrideOutputLayoutOption};
  llvm::StringMap<OutputLayoutOverrideParams> parsedOverride;
};

llvm::cl::opt<std::string> Conv2dConfigOverrideTest::OverrideConv2dConfigOption{
    "override-conv2d-config"};

llvm::cl::opt<std::string> OutputLayoutOverrideTest::OverrideOutputLayoutOption{
    "override-output-layout"};

TEST_F(Conv2dConfigOverrideTest, ParseFullConv2dConfigOverride) {
  std::string arg = "op0="
                    "weights_dtype#bf16:"
                    "activation#relu:"
                    "deallocate_activation#false:"
                    "reallocate_halo_output#true:"
                    "act_block_h_override#32:"
                    "act_block_w_div#1:"
                    "reshard_if_not_optimal#false:"
                    "override_sharding_config#false:"
                    "shard_layout#block_sharded:"
                    "transpose_shards#true:"
                    "output_layout#row_major:"
                    "enable_act_double_buffer#false:"
                    "enable_weights_double_buffer#false";

  bool result = parser.parse(OverrideConv2dConfigOption,
                             "override-conv2d-config", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 1);
  ASSERT_TRUE(parsedOverride.count("op0"));

  const auto &params = parsedOverride["op0"];
  ASSERT_TRUE(params.weightsDtype.has_value());
  ASSERT_EQ(params.weightsDtype.value(), mlir::tt::ttcore::DataType::BFloat16);
  ASSERT_TRUE(params.activation.has_value());
  ASSERT_EQ(params.activation.value(), UnaryOpType::Relu);
  ASSERT_TRUE(params.deallocateActivation.has_value());
  ASSERT_FALSE(params.deallocateActivation.value());
  ASSERT_TRUE(params.reallocateHaloOutput.has_value());
  ASSERT_TRUE(params.reallocateHaloOutput.value());
  ASSERT_TRUE(params.actBlockHOverride.has_value());
  ASSERT_EQ(params.actBlockHOverride.value(), 32);
  ASSERT_TRUE(params.actBlockWDiv.has_value());
  ASSERT_EQ(params.actBlockWDiv.value(), 1);
  ASSERT_TRUE(params.reshardIfNotOptimal.has_value());
  ASSERT_FALSE(params.reshardIfNotOptimal.value());
  ASSERT_TRUE(params.overrideShardingConfig.has_value());
  ASSERT_FALSE(params.overrideShardingConfig.value());
  ASSERT_TRUE(params.shardLayout.has_value());
  ASSERT_EQ(params.shardLayout.value(), TensorMemoryLayout::BlockSharded);
  ASSERT_TRUE(params.transposeShards.has_value());
  ASSERT_TRUE(params.transposeShards.value());
  ASSERT_TRUE(params.outputLayout.has_value());
  ASSERT_EQ(params.outputLayout.value(), Layout::RowMajor);
  ASSERT_TRUE(params.enableActDoubleBuffer.has_value());
  ASSERT_FALSE(params.enableActDoubleBuffer.value());
  ASSERT_TRUE(params.enableWeightsDoubleBuffer.has_value());
  ASSERT_FALSE(params.enableWeightsDoubleBuffer.value());
}

TEST_F(Conv2dConfigOverrideTest, ParsePartialConv2dConfigOverride) {
  std::string arg = "op0=";

  bool result = parser.parse(OverrideConv2dConfigOption,
                             "override-conv2d-config", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 1);
  ASSERT_TRUE(parsedOverride.count("op0"));

  const auto &params = parsedOverride["op0"];
  ASSERT_EQ(params.activation, std::nullopt);
  ASSERT_FALSE(params.weightsDtype.has_value());
  ASSERT_FALSE(params.deallocateActivation.has_value());
  ASSERT_FALSE(params.reallocateHaloOutput.has_value());
  ASSERT_FALSE(params.actBlockHOverride.has_value());
  ASSERT_FALSE(params.actBlockWDiv.has_value());
  ASSERT_FALSE(params.reshardIfNotOptimal.has_value());
  ASSERT_FALSE(params.overrideShardingConfig.has_value());
  ASSERT_FALSE(params.shardLayout.has_value());
  ASSERT_FALSE(params.transposeShards.has_value());
  ASSERT_FALSE(params.outputLayout.has_value());
  ASSERT_FALSE(params.enableActDoubleBuffer.has_value());
  ASSERT_FALSE(params.enableWeightsDoubleBuffer.has_value());
}

TEST_F(Conv2dConfigOverrideTest, ParseMultipleOps) {
  std::string arg = "op0="
                    ","
                    "op1=weights_dtype#bf16:"
                    "activation#relu";

  bool result = parser.parse(OverrideConv2dConfigOption,
                             "override-conv2d-config", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 2);
  ASSERT_TRUE(parsedOverride.count("op0"));
  ASSERT_TRUE(parsedOverride.count("op1"));

  const auto &params0 = parsedOverride["op0"];
  ASSERT_EQ(params0.activation, std::nullopt);

  const auto &params1 = parsedOverride["op1"];
  ASSERT_TRUE(params1.weightsDtype.has_value());
  ASSERT_EQ(params1.weightsDtype.value(), mlir::tt::ttcore::DataType::BFloat16);
  ASSERT_TRUE(params1.activation.has_value());
  ASSERT_EQ(params1.activation.value(), UnaryOpType::Relu);
}

TEST_F(Conv2dConfigOverrideTest, ParseInvalidActivation) {
  std::string arg = "op0=activation#invalid_activation";

  bool result = parser.parse(OverrideConv2dConfigOption,
                             "override-conv2d-config", arg, parsedOverride);
  ASSERT_TRUE(result);
}

TEST_F(Conv2dConfigOverrideTest, ParseInvalidShardLayout) {
  std::string arg = "op0=shard_layout#invalid";

  bool result = parser.parse(OverrideConv2dConfigOption,
                             "override-conv2d-config", arg, parsedOverride);
  ASSERT_TRUE(result);
}

TEST_F(OutputLayoutOverrideTest, ParseFullOutputLayoutOverride) {
  std::string arg = "op1=1x1:dram:interleaved:tile:f32";

  bool result = parser.parse(OverrideOutputLayoutOption,
                             "override-output-layout", arg, parsedOverride);
  ASSERT_FALSE(result);
  ASSERT_EQ(parsedOverride.size(), 1);
  ASSERT_TRUE(parsedOverride.count("op1"));

  const auto &params = parsedOverride["op1"];
  ASSERT_TRUE(params.grid.has_value());
  ASSERT_EQ(params.grid->size(), 2);
  ASSERT_EQ((*params.grid)[0], 1);
  ASSERT_EQ((*params.grid)[1], 1);
  ASSERT_TRUE(params.bufferType.has_value());
  ASSERT_EQ(params.bufferType.value(), BufferType::DRAM);
  ASSERT_TRUE(params.tensorMemoryLayout.has_value());
  ASSERT_EQ(params.tensorMemoryLayout.value(), TensorMemoryLayout::Interleaved);
  ASSERT_TRUE(params.memoryLayout.has_value());
  ASSERT_EQ(params.memoryLayout.value(), Layout::Tile);
  ASSERT_TRUE(params.dataType.has_value());
  ASSERT_EQ(params.dataType.value(), mlir::tt::ttcore::DataType::Float32);
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
  std::string arg = "op1=1x1:dram:interleaved:tile:f32,op2=4x4:l1:block_"
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
  ASSERT_EQ((*params1.grid)[0], 1);
  ASSERT_EQ((*params1.grid)[1], 1);
  ASSERT_TRUE(params1.bufferType.has_value());
  ASSERT_EQ(params1.bufferType.value(), BufferType::DRAM);
  ASSERT_TRUE(params1.tensorMemoryLayout.has_value());
  ASSERT_EQ(params1.tensorMemoryLayout.value(),
            TensorMemoryLayout::Interleaved);
  ASSERT_TRUE(params1.memoryLayout.has_value());
  ASSERT_EQ(params1.memoryLayout.value(), Layout::Tile);
  ASSERT_TRUE(params1.dataType.has_value());
  ASSERT_EQ(params1.dataType.value(), mlir::tt::ttcore::DataType::Float32);

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
  ASSERT_EQ(params2.dataType.value(), mlir::tt::ttcore::DataType::Float16);
}

class TestOptimizerOverrideHandler : public ::testing::Test {

public:
  OptimizerOverridesHandler optimizerOverridesHandler;

  void SetUp() override {}

  llvm::StringMap<InsertMemReconfigParams> createInsertMemReconfig() {

    // struct InsertMemReconfigParams {
    //   SmallVector<int64_t> operandIdxes;
    // };

    llvm::StringMap<InsertMemReconfigParams> insertMemReconfig;

    // Create input layout overrides for 3 input overrides.
    insertMemReconfig["input0"] = createInsertMemReconfigParams();
    insertMemReconfig["input1"] = createInsertMemReconfigParams();
    insertMemReconfig["input2"] = createInsertMemReconfigParams();

    return insertMemReconfig;
  }

  InsertMemReconfigParams createInsertMemReconfigParams() {

    InsertMemReconfigParams insertMemReconfigParams;

    // Create input layout override params for 2 operands.
    // Their operand indexes are 0 and 1, respectively.
    insertMemReconfigParams.operandIdxes.push_back(0);
    insertMemReconfigParams.operandIdxes.push_back(1);

    return insertMemReconfigParams;
  }

  llvm::StringMap<OutputLayoutOverrideParams> createOutputLayoutOverrides() {

    llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides;

    // Create output layout overrides for 3 output overrides.
    outputLayoutOverrides["output0"] = createOutputLayoutOverrideParams_0();
    outputLayoutOverrides["output1"] = createOutputLayoutOverrideParams_1();
    outputLayoutOverrides["output2"] = createOutputLayoutOverrideParams_2();

    return outputLayoutOverrides;
  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_0() {

    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   BufferType;
    //   TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   Layout memoryLayout;             // ROW_MAJOR / TILE
    //   mlir::tt::ttcore::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 0 has
    //      - grid size 1x1,
    //      - buffer type dram
    //      - tensor memory layout interleaved
    //      - memory layout tile
    //      - data type fp16.
    outputLayoutOverrideParams.grid = llvm::SmallVector<int64_t, 2>({1, 1});
    outputLayoutOverrideParams.bufferType = BufferType::DRAM;
    outputLayoutOverrideParams.tensorMemoryLayout =
        TensorMemoryLayout::Interleaved;
    outputLayoutOverrideParams.memoryLayout = Layout::Tile;
    outputLayoutOverrideParams.dataType = mlir::tt::ttcore::DataType::Float16;

    return outputLayoutOverrideParams;
  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_1() {

    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   BufferType;
    //   TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   Layout memoryLayout;             // ROW_MAJOR / TILE
    //   mlir::tt::ttcore::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 1 has
    //      - grid size 8x4,
    //      - buffer type l1
    //      - tensor memory layout block_sharded
    //      - memory layout row_major
    //      - data type fp16.
    outputLayoutOverrideParams.grid = llvm::SmallVector<int64_t, 2>({8, 4});
    outputLayoutOverrideParams.bufferType = BufferType::L1;
    outputLayoutOverrideParams.tensorMemoryLayout =
        TensorMemoryLayout::BlockSharded;
    outputLayoutOverrideParams.memoryLayout = Layout::RowMajor;
    outputLayoutOverrideParams.dataType = mlir::tt::ttcore::DataType::Float16;

    return outputLayoutOverrideParams;
  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_2() {

    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   BufferType;
    //   TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   Layout memoryLayout;             // ROW_MAJOR / TILE
    //   mlir::tt::ttcore::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 2 has
    //      - grid size 1x1,
    //      - buffer type system
    //      - tensor memory layout height_sharded
    //      - memory layout tile
    //      - data type fp16.
    outputLayoutOverrideParams.grid = llvm::SmallVector<int64_t, 2>({1, 1});
    outputLayoutOverrideParams.bufferType = BufferType::SystemMemory;
    outputLayoutOverrideParams.tensorMemoryLayout =
        TensorMemoryLayout::HeightSharded;
    outputLayoutOverrideParams.memoryLayout = Layout::Tile;
    outputLayoutOverrideParams.dataType = mlir::tt::ttcore::DataType::Float16;

    return outputLayoutOverrideParams;
  }

  bool compareInsertMemReconfig(llvm::StringMap<InsertMemReconfigParams> in1,
                                llvm::StringMap<InsertMemReconfigParams> in2) {
    // Check if the sizes of the two input layout overrides are the same.
    if (in1.size() != in2.size()) {
      return false;
    }
    llvm::StringMap<InsertMemReconfigParams>::iterator it1;
    for (it1 = in1.begin(); it1 != in1.end(); it1++) {
      // Check if the two input layout overrides have the same keys.
      llvm::StringMap<InsertMemReconfigParams>::iterator it2 =
          in2.find(it1->getKey());
      if (it2 == in2.end()) {
        return false;
      }
      // Check if the two input layout overrides have the same values.
      // The structure InsertMemReconfigParams has overloaded operators for ==
      // and !=, so we can compare the objects in this way.
      if (it1->getValue() != it2->getValue()) {
        return false;
      }
    }
    return true;
  }

  bool compareOutputLayoutOverrides(
      llvm::StringMap<OutputLayoutOverrideParams> out1,
      llvm::StringMap<OutputLayoutOverrideParams> out2) {
    // Check if the sizes of the two output layout overrides are the same.
    if (out1.size() != out2.size()) {
      return false;
    }
    llvm::StringMap<OutputLayoutOverrideParams>::iterator it1;
    for (it1 = out1.begin(); it1 != out1.end(); it1++) {
      // Check if the two output layout overrides have the same keys.
      llvm::StringMap<OutputLayoutOverrideParams>::iterator it2 =
          out2.find(it1->getKey());
      if (it2 == out2.end()) {
        return false;
      }
      // Check if the two output layout overrides have the same values.
      // The structure OutputLayoutOverrideParams has overloaded operators for
      // == and !=, so we can compare the objects in this way.
      if (it1->getValue() != it2->getValue()) {
        return false;
      }
    }
    return true;
  }

  void TearDown() override {}
};

// Test the setEnableOptimizer method
TEST_F(TestOptimizerOverrideHandler, TestSetOptimizerPass) {

  optimizerOverridesHandler.setEnableOptimizer(true);
  ASSERT_TRUE(optimizerOverridesHandler.getEnableOptimizer());

  optimizerOverridesHandler.setEnableOptimizer(false);
  ASSERT_FALSE(optimizerOverridesHandler.getEnableOptimizer());
}

// Test the setMemoryConfig method
TEST_F(TestOptimizerOverrideHandler, TestSetMemoryConfig) {

  optimizerOverridesHandler.setMemoryReconfig(true);
  ASSERT_TRUE(optimizerOverridesHandler.getMemoryReconfig());

  optimizerOverridesHandler.setMemoryReconfig(false);
  ASSERT_FALSE(optimizerOverridesHandler.getMemoryReconfig());
}

// Test the setMemoryLayoutAnalysis method
TEST_F(TestOptimizerOverrideHandler, TestSetMemoryLayoutAnalysis) {

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysis(true);
  ASSERT_TRUE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysis());

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysis(false);
  ASSERT_FALSE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysis());
}

// Test the setEnableL1InterleavedFallbackAnalysis method
TEST_F(TestOptimizerOverrideHandler,
       TestSetEnableL1InterleavedFallbackAnalysis) {

  optimizerOverridesHandler.setEnableL1InterleavedFallbackAnalysis(true);
  ASSERT_TRUE(
      optimizerOverridesHandler.getEnableL1InterleavedFallbackAnalysis());

  optimizerOverridesHandler.setEnableL1InterleavedFallbackAnalysis(false);
  ASSERT_FALSE(
      optimizerOverridesHandler.getEnableL1InterleavedFallbackAnalysis());
}

// Test the setEnableMemoryLayoutAnalysisPolicy method
TEST_F(TestOptimizerOverrideHandler, TestSetEnableMemoryLayoutAnalysisPolicy) {

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysisPolicy(true);
  ASSERT_TRUE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysisPolicy());

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysisPolicy(false);
  ASSERT_FALSE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysisPolicy());
}

// Test the setMemoryLayoutAnalysisPolicy method
TEST_F(TestOptimizerOverrideHandler, TestSetMemoryLayoutAnalysisPolicy) {

  optimizerOverridesHandler.setMemoryLayoutAnalysisPolicy(
      mlir::tt::MemoryLayoutAnalysisPolicyType::DFSharding);
  ASSERT_EQ(optimizerOverridesHandler.getMemoryLayoutAnalysisPolicy(),
            mlir::tt::MemoryLayoutAnalysisPolicyType::DFSharding);

  optimizerOverridesHandler.setMemoryLayoutAnalysisPolicy(
      mlir::tt::MemoryLayoutAnalysisPolicyType::GreedyL1Interleaved);
  ASSERT_EQ(optimizerOverridesHandler.getMemoryLayoutAnalysisPolicy(),
            mlir::tt::MemoryLayoutAnalysisPolicyType::GreedyL1Interleaved);
}

// Test the setInsertMemReconfig method
TEST_F(TestOptimizerOverrideHandler, TestSetInsertMemReconfig) {

  llvm::StringMap<InsertMemReconfigParams> insertMemReconfig =
      createInsertMemReconfig();

  optimizerOverridesHandler.setInsertMemReconfig(insertMemReconfig);
  ASSERT_TRUE(compareInsertMemReconfig(
      optimizerOverridesHandler.getInsertMemReconfig(), insertMemReconfig));
}

// Test the setOutputLayoutOverrides method
TEST_F(TestOptimizerOverrideHandler, TestSetOutputLayoutOverrides) {

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides =
      createOutputLayoutOverrides();

  optimizerOverridesHandler.setOutputLayoutOverrides(outputLayoutOverrides);
  ASSERT_TRUE(compareOutputLayoutOverrides(
      optimizerOverridesHandler.getOutputLayoutOverrides(),
      outputLayoutOverrides));
}

// Test the addinsertMemReconfig method passing the whole object
TEST_F(TestOptimizerOverrideHandler, TestAddinsertMemReconfigObject) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the first function, which takes the whole object as a
  // parameter.

  llvm::StringMap<InsertMemReconfigParams> insertMemReconfig =
      createInsertMemReconfig();

  optimizerOverridesHandler.addInsertMemReconfig(
      "input0", createInsertMemReconfigParams());
  optimizerOverridesHandler.addInsertMemReconfig(
      "input1", createInsertMemReconfigParams());
  optimizerOverridesHandler.addInsertMemReconfig(
      "input2", createInsertMemReconfigParams());

  ASSERT_TRUE(compareInsertMemReconfig(
      optimizerOverridesHandler.getInsertMemReconfig(), insertMemReconfig));
}

// Test the addInsertMemReconfig method passing the individual parameters
TEST_F(TestOptimizerOverrideHandler, TestAddInsertMemReconfigParams) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the second function, which takes the individual parameters.

  llvm::StringMap<InsertMemReconfigParams> insertMemReconfig =
      createInsertMemReconfig();

  llvm::SmallVector<int64_t> operandIdxes1 = {0, 1};
  llvm::SmallVector<int64_t> operandIdxes2 = {0, 1};
  llvm::SmallVector<int64_t> operandIdxes3 = {0, 1};

  optimizerOverridesHandler.addInsertMemReconfig("input0", operandIdxes1);
  optimizerOverridesHandler.addInsertMemReconfig("input1", operandIdxes2);
  optimizerOverridesHandler.addInsertMemReconfig("input2", operandIdxes3);

  ASSERT_TRUE(compareInsertMemReconfig(
      optimizerOverridesHandler.getInsertMemReconfig(), insertMemReconfig));
}

// Test the addOutputLayoutOverride method passing the whole object
TEST_F(TestOptimizerOverrideHandler, TestAddOutputLayoutOverrideObject) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the first function, which takes the whole object as a
  // parameter.

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides =
      createOutputLayoutOverrides();

  optimizerOverridesHandler.addOutputLayoutOverride(
      "output0", createOutputLayoutOverrideParams_0());
  optimizerOverridesHandler.addOutputLayoutOverride(
      "output1", createOutputLayoutOverrideParams_1());
  optimizerOverridesHandler.addOutputLayoutOverride(
      "output2", createOutputLayoutOverrideParams_2());

  ASSERT_TRUE(compareOutputLayoutOverrides(
      optimizerOverridesHandler.getOutputLayoutOverrides(),
      outputLayoutOverrides));
}

// Test the addOutputLayoutOverride method passing the individual parameters
TEST_F(TestOptimizerOverrideHandler, TestAddOutputLayoutOverrideParams) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the second function, which takes the individual parameters.

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides =
      createOutputLayoutOverrides();

  llvm::SmallVector<int64_t> grid1 = {1, 1};
  llvm::SmallVector<int64_t> grid2 = {8, 4};
  llvm::SmallVector<int64_t> grid3 = {1, 1};

  optimizerOverridesHandler.addOutputLayoutOverride(
      "output0", grid1, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      Layout::Tile, mlir::tt::ttcore::DataType::Float16);
  optimizerOverridesHandler.addOutputLayoutOverride(
      "output1", grid2, BufferType::L1, TensorMemoryLayout::BlockSharded,
      Layout::RowMajor, mlir::tt::ttcore::DataType::Float16);
  optimizerOverridesHandler.addOutputLayoutOverride(
      "output2", grid3, BufferType::SystemMemory,
      TensorMemoryLayout::HeightSharded, Layout::Tile,
      mlir::tt::ttcore::DataType::Float16);

  ASSERT_TRUE(compareOutputLayoutOverrides(
      optimizerOverridesHandler.getOutputLayoutOverrides(),
      outputLayoutOverrides));
}

// Test the setSystemDescPath method
TEST_F(TestOptimizerOverrideHandler, TestSetSystemDescPath) {

  optimizerOverridesHandler.setSystemDescPath("system_desc_path");
  ASSERT_EQ(optimizerOverridesHandler.getSystemDescPath(), "system_desc_path");
}

// Test the setMaxLegalLayouts method
TEST_F(TestOptimizerOverrideHandler, TestSetMaxLegalLayouts) {

  optimizerOverridesHandler.setMaxLegalLayouts(10);
  ASSERT_EQ(optimizerOverridesHandler.getMaxLegalLayouts(), 10);
}

// Test the setMeshShape method
TEST_F(TestOptimizerOverrideHandler, TestSetMeshShape) {

  std::vector<int64_t> meshShape;
  meshShape.push_back(1);
  meshShape.push_back(2);

  optimizerOverridesHandler.setMeshShape(meshShape);
  ASSERT_EQ(optimizerOverridesHandler.getMeshShape()[0], meshShape[0]);
  ASSERT_EQ(optimizerOverridesHandler.getMeshShape()[1], meshShape[1]);
}

// Test the setTensorL1UsageCap method
TEST_F(TestOptimizerOverrideHandler, TestSetTensorL1UsageCap) {

  optimizerOverridesHandler.setTensorL1UsageCap(1);
  ASSERT_EQ(optimizerOverridesHandler.getTensorL1UsageCap(), 1);
}

// Test the toString method
TEST_F(TestOptimizerOverrideHandler, TestToString) {

  std::string options;
  options +=
      "enable-optimizer=true "; // The optimizer pass is enabled by default.
  options += "memreconfig-enabled=true ";
  options += "memory-layout-analysis-enabled=true ";
  options += "l1-interleaved-fallback-analysis-enabled=true ";
  options += "insert-memreconfig=add_0_1_2=0 ";
  options +=
      "override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32";

  llvm::SmallVector<int64_t> operandIdxes = {0};
  llvm::SmallVector<int64_t> grid = {1, 1};

  optimizerOverridesHandler.setEnableOptimizer(true);
  optimizerOverridesHandler.setEnableMemoryLayoutAnalysis(true);
  optimizerOverridesHandler.setEnableL1InterleavedFallbackAnalysis(true);
  optimizerOverridesHandler.setMemoryReconfig(true);
  optimizerOverridesHandler.addInsertMemReconfig("add_0_1_2", operandIdxes);
  optimizerOverridesHandler.addOutputLayoutOverride(
      "add_1_2", grid, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      Layout::RowMajor, mlir::tt::ttcore::DataType::Float32);

  ASSERT_EQ(optimizerOverridesHandler.toString(), options);
}
