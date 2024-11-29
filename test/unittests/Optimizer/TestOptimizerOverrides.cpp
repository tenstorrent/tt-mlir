// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

using namespace mlir::tt::ttnn;

class TestOptimizerOverrides : public ::testing::Test {

public:
  OptimizerOverridesHandler optimizerOverridesHandler;

  void SetUp() override {}

  llvm::StringMap<InputLayoutOverrideParams> createInputLayoutOverrides() {

    // struct InputLayoutOverrideParams {
    //   SmallVector<int64_t> operandIdxes;
    // };

    llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides;

    // Create input layout overrides for 3 input overrides.
    inputLayoutOverrides["input0"] = createInputLayoutOverrideParams();
    inputLayoutOverrides["input1"] = createInputLayoutOverrideParams();
    inputLayoutOverrides["input2"] = createInputLayoutOverrideParams();

    return inputLayoutOverrides;
  }

  InputLayoutOverrideParams createInputLayoutOverrideParams() {

    InputLayoutOverrideParams inputLayoutOverrideParams;

    // Create input layout override params for 2 operands.
    // Their operand indexes are 0 and 1, respectively.
    inputLayoutOverrideParams.operandIdxes.push_back(0);
    inputLayoutOverrideParams.operandIdxes.push_back(1);

    return inputLayoutOverrideParams;
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
    //   mlir::tt::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 0 has
    //      - grid size 2x2,
    //      - buffer type dram
    //      - tensor memory layout interleaved
    //      - memory layout tile
    //      - data type fp16.
    outputLayoutOverrideParams.grid.push_back(2);
    outputLayoutOverrideParams.grid.push_back(2);
    outputLayoutOverrideParams.bufferType = BufferType::DRAM;
    outputLayoutOverrideParams.tensorMemoryLayout =
        TensorMemoryLayout::Interleaved;
    outputLayoutOverrideParams.memoryLayout = Layout::Tile;
    outputLayoutOverrideParams.dataType = mlir::tt::DataType::Float16;

    return outputLayoutOverrideParams;
  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_1() {

    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   BufferType;
    //   TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   Layout memoryLayout;             // ROW_MAJOR / TILE
    //   mlir::tt::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 1 has
    //      - grid size 8x4,
    //      - buffer type l1
    //      - tensor memory layout block_sharded
    //      - memory layout row_major
    //      - data type fp16.
    outputLayoutOverrideParams.grid.push_back(8);
    outputLayoutOverrideParams.grid.push_back(4);
    outputLayoutOverrideParams.bufferType = BufferType::L1;
    outputLayoutOverrideParams.tensorMemoryLayout =
        TensorMemoryLayout::BlockSharded;
    outputLayoutOverrideParams.memoryLayout = Layout::RowMajor;
    outputLayoutOverrideParams.dataType = mlir::tt::DataType::Float16;

    return outputLayoutOverrideParams;
  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_2() {

    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   BufferType;
    //   TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   Layout memoryLayout;             // ROW_MAJOR / TILE
    //   mlir::tt::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 2 has
    //      - grid size 3x6,
    //      - buffer type system
    //      - tensor memory layout height_sharded
    //      - memory layout tile
    //      - data type fp16.
    outputLayoutOverrideParams.grid.push_back(3);
    outputLayoutOverrideParams.grid.push_back(6);
    outputLayoutOverrideParams.bufferType = BufferType::SystemMemory;
    outputLayoutOverrideParams.tensorMemoryLayout =
        TensorMemoryLayout::HeightSharded;
    outputLayoutOverrideParams.memoryLayout = Layout::Tile;
    outputLayoutOverrideParams.dataType = mlir::tt::DataType::Float16;

    return outputLayoutOverrideParams;
  }

  bool
  compareInputLayoutOverrides(llvm::StringMap<InputLayoutOverrideParams> in1,
                              llvm::StringMap<InputLayoutOverrideParams> in2) {
    // Check if the sizes of the two input layout overrides are the same.
    if (in1.size() != in2.size()) {
      return false;
    }
    llvm::StringMap<InputLayoutOverrideParams>::iterator it1;
    for (it1 = in1.begin(); it1 != in1.end(); it1++) {
      // Check if the two input layout overrides have the same keys.
      llvm::StringMap<InputLayoutOverrideParams>::iterator it2 =
          in2.find(it1->getKey());
      if (it2 == in2.end()) {
        return false;
      }
      // Check if the two input layout overrides have the same values.
      // The structure InputLayoutOverrideParams has overloaded operators for ==
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
TEST_F(TestOptimizerOverrides, TestSetOptimizerPass) {

  optimizerOverridesHandler.setEnableOptimizer(true);
  ASSERT_TRUE(optimizerOverridesHandler.getEnableOptimizer());

  optimizerOverridesHandler.setEnableOptimizer(false);
  ASSERT_FALSE(optimizerOverridesHandler.getEnableOptimizer());
}

// Test the setMemoryConfig method
TEST_F(TestOptimizerOverrides, TestSetMemoryConfig) {

  optimizerOverridesHandler.setMemoryReconfig(true);
  ASSERT_TRUE(optimizerOverridesHandler.getMemoryReconfig());

  optimizerOverridesHandler.setMemoryReconfig(false);
  ASSERT_FALSE(optimizerOverridesHandler.getMemoryReconfig());
}

// Test the setMemoryLayoutAnalysis method
TEST_F(TestOptimizerOverrides, TestSetMemoryLayoutAnalysis) {

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysis(true);
  ASSERT_TRUE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysis());

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysis(false);
  ASSERT_FALSE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysis());
}

// Test the setEnableMemoryLayoutAnalysisPolicy method
TEST_F(TestOptimizerOverrides, TestSetEnableMemoryLayoutAnalysisPolicy) {

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysisPolicy(true);
  ASSERT_TRUE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysisPolicy());

  optimizerOverridesHandler.setEnableMemoryLayoutAnalysisPolicy(false);
  ASSERT_FALSE(optimizerOverridesHandler.getEnableMemoryLayoutAnalysisPolicy());
}

// Test the setMemoryLayoutAnalysisPolicy method
TEST_F(TestOptimizerOverrides, TestSetMemoryLayoutAnalysisPolicy) {

  optimizerOverridesHandler.setMemoryLayoutAnalysisPolicy(
      mlir::tt::MemoryLayoutAnalysisPolicyType::DFSharding);
  ASSERT_EQ(optimizerOverridesHandler.getMemoryLayoutAnalysisPolicy(),
            mlir::tt::MemoryLayoutAnalysisPolicyType::DFSharding);

  optimizerOverridesHandler.setMemoryLayoutAnalysisPolicy(
      mlir::tt::MemoryLayoutAnalysisPolicyType::L1Interleaved);
  ASSERT_EQ(optimizerOverridesHandler.getMemoryLayoutAnalysisPolicy(),
            mlir::tt::MemoryLayoutAnalysisPolicyType::L1Interleaved);
}

// Test the setInputLayoutOverrides method
TEST_F(TestOptimizerOverrides, TestSetInputLayoutOverrides) {

  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides =
      createInputLayoutOverrides();

  optimizerOverridesHandler.setInputLayoutOverrides(inputLayoutOverrides);
  ASSERT_TRUE(compareInputLayoutOverrides(
      optimizerOverridesHandler.getInputLayoutOverrides(),
      inputLayoutOverrides));
}

// Test the setOutputLayoutOverrides method
TEST_F(TestOptimizerOverrides, TestSetOutputLayoutOverrides) {

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides =
      createOutputLayoutOverrides();

  optimizerOverridesHandler.setOutputLayoutOverrides(outputLayoutOverrides);
  ASSERT_TRUE(compareOutputLayoutOverrides(
      optimizerOverridesHandler.getOutputLayoutOverrides(),
      outputLayoutOverrides));
}

// Test the addInputLayoutOverride method passing the whole object
TEST_F(TestOptimizerOverrides, TestAddInputLayoutOverrideObject) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the first function, which takes the whole object as a
  // parameter.

  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides =
      createInputLayoutOverrides();

  optimizerOverridesHandler.addInputLayoutOverride(
      "input0", createInputLayoutOverrideParams());
  optimizerOverridesHandler.addInputLayoutOverride(
      "input1", createInputLayoutOverrideParams());
  optimizerOverridesHandler.addInputLayoutOverride(
      "input2", createInputLayoutOverrideParams());

  ASSERT_TRUE(compareInputLayoutOverrides(
      optimizerOverridesHandler.getInputLayoutOverrides(),
      inputLayoutOverrides));
}

// Test the addInputLayoutOverride method passing the individual parameters
TEST_F(TestOptimizerOverrides, TestAddInputLayoutOverrideParams) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the second function, which takes the individual parameters.

  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides =
      createInputLayoutOverrides();

  llvm::SmallVector<int64_t> operandIdxes1 = {0, 1};
  llvm::SmallVector<int64_t> operandIdxes2 = {0, 1};
  llvm::SmallVector<int64_t> operandIdxes3 = {0, 1};

  optimizerOverridesHandler.addInputLayoutOverride("input0", operandIdxes1);
  optimizerOverridesHandler.addInputLayoutOverride("input1", operandIdxes2);
  optimizerOverridesHandler.addInputLayoutOverride("input2", operandIdxes3);

  ASSERT_TRUE(compareInputLayoutOverrides(
      optimizerOverridesHandler.getInputLayoutOverrides(),
      inputLayoutOverrides));
}

// Test the addOutputLayoutOverride method passing the whole object
TEST_F(TestOptimizerOverrides, TestAddOutputLayoutOverrideObject) {

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
TEST_F(TestOptimizerOverrides, TestAddOutputLayoutOverrideParams) {

  // This method is implemented across two functions in the
  // OptimizerOverridesHandler class. The first function takes the whole object
  // as a parameter, while the second function takes the individual parameters.

  // Here, we test the second function, which takes the individual parameters.

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides =
      createOutputLayoutOverrides();

  llvm::SmallVector<int64_t> grid1 = {2, 2};
  llvm::SmallVector<int64_t> grid2 = {8, 4};
  llvm::SmallVector<int64_t> grid3 = {3, 6};

  optimizerOverridesHandler.addOutputLayoutOverride(
      "output0", grid1, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      Layout::Tile, mlir::tt::DataType::Float16);
  optimizerOverridesHandler.addOutputLayoutOverride(
      "output1", grid2, BufferType::L1, TensorMemoryLayout::BlockSharded,
      Layout::RowMajor, mlir::tt::DataType::Float16);
  optimizerOverridesHandler.addOutputLayoutOverride(
      "output2", grid3, BufferType::SystemMemory,
      TensorMemoryLayout::HeightSharded, Layout::Tile,
      mlir::tt::DataType::Float16);

  ASSERT_TRUE(compareOutputLayoutOverrides(
      optimizerOverridesHandler.getOutputLayoutOverrides(),
      outputLayoutOverrides));
}

// Test the setSystemDescPath method
TEST_F(TestOptimizerOverrides, TestSetSystemDescPath) {

  optimizerOverridesHandler.setSystemDescPath("system_desc_path");
  ASSERT_EQ(optimizerOverridesHandler.getSystemDescPath(), "system_desc_path");
}

// Test the setMaxLegalLayouts method
TEST_F(TestOptimizerOverrides, TestSetMaxLegalLayouts) {

  optimizerOverridesHandler.setMaxLegalLayouts(10);
  ASSERT_EQ(optimizerOverridesHandler.getMaxLegalLayouts(), 10);
}

// Test the setMeshShape method
TEST_F(TestOptimizerOverrides, TestSetMeshShape) {

  std::vector<int64_t> meshShape;
  meshShape.push_back(1);
  meshShape.push_back(2);

  optimizerOverridesHandler.setMeshShape(meshShape);
  ASSERT_EQ(optimizerOverridesHandler.getMeshShape()[0], meshShape[0]);
  ASSERT_EQ(optimizerOverridesHandler.getMeshShape()[1], meshShape[1]);
}

// Test the toString method
TEST_F(TestOptimizerOverrides, TestToString) {

  std::string options;
  options +=
      "enable-optimizer=true "; // The optimizer pass is enabled by default.
  options += "memreconfig-enabled=true ";
  options += "memory-layout-analysis-enabled=true ";
  options += "insert-memreconfig=add_0_1_2=0 ";
  options +=
      "override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32";

  llvm::SmallVector<int64_t> operandIdxes = {0};
  llvm::SmallVector<int64_t> grid = {1, 1};

  optimizerOverridesHandler.setEnableOptimizer(true);
  optimizerOverridesHandler.setEnableMemoryLayoutAnalysis(true);
  optimizerOverridesHandler.setMemoryReconfig(true);
  optimizerOverridesHandler.addInputLayoutOverride("add_0_1_2", operandIdxes);
  optimizerOverridesHandler.addOutputLayoutOverride(
      "add_1_2", grid, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      Layout::RowMajor, mlir::tt::DataType::Float32);

  ASSERT_EQ(optimizerOverridesHandler.toString(), options);
}
