#include <gtest/gtest.h>

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"


class TestOptimizerOverrides : public ::testing::Test {

public:
  
  OptimizerOverridesHandler *optimizerOverridesHandler;

  void SetUp() override {
    optimizerOverridesHandler = new OptimizerOverridesHandler();
  }

  llvm::StringMap<InputLayoutOverrideParams> createInputLayoutOverrides() {

    // struct InputLayoutOverrideParams {
    //   SmallVector<int64_t> operandIdxes;
    // };

    llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides;

    // Create input layout overrides for 3 input overrides.
    inputLayoutOverrides.insert("input0", createInputLayoutOverrideParams());
    inputLayoutOverrides.insert("input1", createInputLayoutOverrideParams());
    inputLayoutOverrides.insert("input2", createInputLayoutOverrideParams());

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
    outputLayoutOverrides.insert("output0", createOutputLayoutOverrideParams_0());
    outputLayoutOverrides.insert("output1", createOutputLayoutOverrideParams_1());
    outputLayoutOverrides.insert("output2", createOutputLayoutOverrideParams_2());

    return outputLayoutOverrides;

  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_0() {

    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   tt::MemorySpace memorySpace;
    //   tt::TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   tt::ttnn::Layout memoryLayout;             // ROW_MAJOR / TILE
    //   tt::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 0 has
    //      - grid size 2x2, 
    //      - memory space dram 
    //      - tensor memory layout interleaved
    //      - memory layout tile 
    //      - data type fp16.
    outputLayoutOverrides["output0"].grid.push_back(2);
    outputLayoutOverrides["output0"].grid.push_back(2);
    outputLayoutOverrides["output0"].memorySpace = tt::MemorySpace::DeviceDRAM;
    outputLayoutOverrides["output0"].tensorMemoryLayout = tt::TensorMemoryLayout::Interleaved;
    outputLayoutOverrides["output0"].memoryLayout = tt::ttnn::Layout::Tile;
    outputLayoutOverrides["output0"].dataType = tt::DataType::Float16;

    return outputLayoutOverrideParams;

  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_1() {
    
    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   tt::MemorySpace memorySpace;
    //   tt::TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   tt::ttnn::Layout memoryLayout;             // ROW_MAJOR / TILE
    //   tt::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 1 has
    //      - grid size 8x4,
    //      - memory space l1
    //      - tensor memory layout block_sharded
    //      - memory layout row_major
    //      - data type fp16.
    outputLayoutOverrides["output1"].grid.push_back(8);
    outputLayoutOverrides["output1"].grid.push_back(4);
    outputLayoutOverrides["output1"].memorySpace = tt::MemorySpace::DeviceL1;
    outputLayoutOverrides["output1"].tensorMemoryLayout = tt::TensorMemoryLayout::BlockSharded;
    outputLayoutOverrides["output1"].memoryLayout = tt::ttnn::Layout::RowMajor;
    outputLayoutOverrides["output1"].dataType = tt::DataType::Float16;

    return outputLayoutOverrideParams;
    
  }

  OutputLayoutOverrideParams createOutputLayoutOverrideParams_2() {
    
    // struct OutputLayoutOverrideParams {
    //   SmallVector<int64_t, 2> grid;
    //   tt::MemorySpace memorySpace;
    //   tt::TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
    //   tt::ttnn::Layout memoryLayout;             // ROW_MAJOR / TILE
    //   tt::DataType dataType;
    // };

    OutputLayoutOverrideParams outputLayoutOverrideParams;

    // Output 2 has
    //      - grid size 3x6,
    //      - memory space system
    //      - tensor memory layout height_sharded
    //      - memory layout tile
    //      - data type fp16.
    outputLayoutOverrides["output2"].grid.push_back(3);
    outputLayoutOverrides["output2"].grid.push_back(6);
    outputLayoutOverrides["output2"].memorySpace = tt::MemorySpace::System;
    outputLayoutOverrides["output2"].tensorMemoryLayout = tt::TensorMemoryLayout::HeightSharded;
    outputLayoutOverrides["output2"].memoryLayout = tt::ttnn::Layout::Tile;
    outputLayoutOverrides["output2"].dataType = tt::DataType::Float16;

    return outputLayoutOverrideParams;
    
  }

  bool compareInputLayoutOverrides(llvm::StringMap<InputLayoutOverrideParams> in1, llvm::StringMap<InputLayoutOverrideParams> in2) {
    // Check if the sizes of the two input layout overrides are the same.
    if (in1.size() != in2.size()) {
      return false;
    }
    llvm::StringMap<InputLayoutOverrideParams>::iterator it1;
    for (it1 = in1.begin(); it1 != in1.end(); it1++) {
      // Check if the two input layout overrides have the same keys.
      llvm::StringMap<InputLayoutOverrideParams>::iterator it2 = in2.find(it1->first);
      if (it2 == in2.end()) {
        return false;
      }
      // Check if the two input layout overrides have the same values.
      // The structure InputLayoutOverrideParams has overloaded operators for == and !=, so we can compare the objects in this way.
      if (it1->second != it2->second) {
        return false;
      }
    }
    return true;
  }

  bool compareOutputLayoutOverrides(llvm::StringMap<OutputLayoutOverrideParams> out1, llvm::StringMap<OutputLayoutOverrideParams> out2) {
    // Check if the sizes of the two output layout overrides are the same.
    if (out1.size() != out2.size()) {
      return false;
    }
    llvm::StringMap<OutputLayoutOverrideParams>::iterator it1;
    for (it1 = out1.begin(); it1 != out1.end(); it1++) {
      // Check if the two output layout overrides have the same keys.
      llvm::StringMap<OutputLayoutOverrideParams>::iterator it2 = out2.find(it1->first);
      if (it2 == out2.end()) {
        return false;
      }
      // Check if the two output layout overrides have the same values.
      // The structure OutputLayoutOverrideParams has overloaded operators for == and !=, so we can compare the objects in this way.
      if (it1->second != it2->second) {
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    delete optimizerOverridesHandler;
  }

};

// Test the setOptimizerPass method
TEST_F(TestOptimizerOverrides, TestSetOptimizerPass) {
  
  optimizerOverridesHandler->setOptimizerPass(true);
  ASSERT_TRUE(optimizerOverridesHandler->getOptimizerPass());
  
  optimizerOverridesHandler->setOptimizerPass(false);
  ASSERT_FALSE(optimizerOverridesHandler->getOptimizerPass());

}

// Test the setMemoryConfig method
TEST_F(TestOptimizerOverrides, TestSetMemoryConfig) {
  
  optimizerOverridesHandler->setMemoryConfig(true);
  ASSERT_TRUE(optimizerOverridesHandler->getMemoryConfig());
  
  optimizerOverridesHandler->setMemoryConfig(false);
  ASSERT_FALSE(optimizerOverridesHandler->getMemoryConfig());

}

// Test the setMemoryLayoutAnalysis method
TEST_F(TestOptimizerOverrides, TestSetMemoryLayoutAnalysis) {
  
  optimizerOverridesHandler->setMemoryLayoutAnalysis(true);
  ASSERT_TRUE(optimizerOverridesHandler->getMemoryLayoutAnalysis());
  
  optimizerOverridesHandler->setMemoryLayoutAnalysis(false);
  ASSERT_FALSE(optimizerOverridesHandler->getMemoryLayoutAnalysis());

}

// Test the setMemoryLayoutAnalysisPolicy method
TEST_F(TestOptimizerOverrides, TestSetMemoryLayoutAnalysisPolicy) {
  
  optimizerOverridesHandler->setMemoryLayoutAnalysisPolicy(tt::MemoryLayoutAnalysisPolicyType::DFSharding);
  ASSERT_EQ(optimizerOverridesHandler->getMemoryLayoutAnalysisPolicy(), tt::MemoryLayoutAnalysisPolicyType::DFSharding);
  
  optimizerOverridesHandler->setMemoryLayoutAnalysisPolicy(tt::MemoryLayoutAnalysisPolicyType::L1Interleaved);
  ASSERT_EQ(optimizerOverridesHandler->getMemoryLayoutAnalysisPolicy(), tt::MemoryLayoutAnalysisPolicyType::L1Interleaved);

}

// Test the setInputLayoutOverrides method
TEST_F(TestOptimizerOverrides, TestSetInputLayoutOverrides) {

  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides = optimizerOverridesHandler->createInputLayoutOverrides()

  optimizerOverridesHandler->setInputLayoutOverrides(inputLayoutOverrides);
  ASSERT_TRUE(compareInputLayoutOverrides(optimizerOverridesHandler->getInputLayoutOverrides(), inputLayoutOverrides));

}

// Test the setOutputLayoutOverrides method
TEST_F(TestOptimizerOverrides, TestSetOutputLayoutOverrides) {

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides = optimizerOverridesHandler->createOutputLayoutOverrides()

  optimizerOverridesHandler->setOutputLayoutOverrides(outputLayoutOverrides);
  ASSERT_TRUE(compareOutputLayoutOverrides(optimizerOverridesHandler->getOutputLayoutOverrides(), outputLayoutOverrides));

}

// Test the addInputLayoutOverride method passing the whole object
TEST_F(TestOptimizerOverrides, TestAddInputLayoutOverrideObject) {

  // This method is implemented across two functions in the OptimizerOverridesHandler class.
  // The first function takes the whole object as a parameter, while the second function takes the individual parameters.

  // Here, we test the first function, which takes the whole object as a parameter.

  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides = createInputLayoutOverrides();

  optimizerOverridesHandler->addInputLayoutOverride("input0", createInputLayoutOverrideParams());
  optimizerOverridesHandler->addInputLayoutOverride("input1", createInputLayoutOverrideParams());
  optimizerOverridesHandler->addInputLayoutOverride("input2", createInputLayoutOverrideParams());

  ASSERT_TRUE(compareInputLayoutOverrides(optimizerOverridesHandler->getInputLayoutOverrides(), inputLayoutOverrides));

}

// Test the addInputLayoutOverride method passing the individual parameters
TEST_F(TestOptimizerOverrides, TestAddInputLayoutOverrideParams) {

  // This method is implemented across two functions in the OptimizerOverridesHandler class.
  // The first function takes the whole object as a parameter, while the second function takes the individual parameters.

  // Here, we test the second function, which takes the individual parameters.

  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides = createInputLayoutOverrides();

  optimizerOverridesHandler->addInputLayoutOverride("input0", { 0, 1 });
  optimizerOverridesHandler->addInputLayoutOverride("input1", { 0, 1 });
  optimizerOverridesHandler->addInputLayoutOverride("input2", { 0, 1 });

  ASSERT_TRUE(compareInputLayoutOverrides(optimizerOverridesHandler->getInputLayoutOverrides(), inputLayoutOverrides));

}

// Test the addOutputLayoutOverride method passing the whole object
TEST_F(TestOptimizerOverrides, TestAddOutputLayoutOverrideObject) {

  // This method is implemented across two functions in the OptimizerOverridesHandler class.
  // The first function takes the whole object as a parameter, while the second function takes the individual parameters.

  // Here, we test the first function, which takes the whole object as a parameter.

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides = createOutputLayoutOverrides();

  optimizerOverridesHandler->addOutputLayoutOverride("output0", createOutputLayoutOverrideParams_0());
  optimizerOverridesHandler->addOutputLayoutOverride("output1", createOutputLayoutOverrideParams_1());
  optimizerOverridesHandler->addOutputLayoutOverride("output2", createOutputLayoutOverrideParams_2());

  ASSERT_TRUE(compareOutputLayoutOverrides(optimizerOverridesHandler->getOutputLayoutOverrides(), outputLayoutOverrides));

}

// Test the addOutputLayoutOverride method passing the individual parameters
TEST_F(TestOptimizerOverrides, TestAddOutputLayoutOverrideParams) {

  // This method is implemented across two functions in the OptimizerOverridesHandler class.
  // The first function takes the whole object as a parameter, while the second function takes the individual parameters.

  // Here, we test the second function, which takes the individual parameters.

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides = createOutputLayoutOverrides();

  optimizerOverridesHandler->addOutputLayoutOverride("output0", { 2, 2 }, tt::MemorySpace::DRAM, tt::TensorMemoryLayout::INTERLEAVED, tt::ttnn::Layout::TILE, tt::DataType::FP16);
  optimizerOverridesHandler->addOutputLayoutOverride("output1", { 8, 4 }, tt::MemorySpace::L1, tt::TensorMemoryLayout::BLOCK_SHARDED, tt::ttnn::Layout::ROW_MAJOR, tt::DataType::FP16);
  optimizerOverridesHandler->addOutputLayoutOverride("output2", { 3, 6 }, tt::MemorySpace::SYSTEM, tt::TensorMemoryLayout::HEIGHT_SHARDED, tt::ttnn::Layout::TILE, tt::DataType::FP16);

  ASSERT_TRUE(compareOutputLayoutOverrides(optimizerOverridesHandler->getOutputLayoutOverrides(), outputLayoutOverrides));

}

// Test the setSystemDescPath method
TEST_F(TestOptimizerOverrides, TestSetSystemDescPath) {
  
  optimizerOverridesHandler->setSystemDescPath("system_desc_path");
  ASSERT_EQ(optimizerOverridesHandler->getSystemDescPath(), "system_desc_path");

}

// Test the setMaxLegalLayouts method
TEST_F(TestOptimizerOverrides, TestSetMaxLegalLayouts) {
  
  optimizerOverridesHandler->setMaxLegalLayouts(10);
  ASSERT_EQ(optimizerOverridesHandler->getMaxLegalLayouts(), 10);

}

// Test the setMeshShape method
TEST_F(TestOptimizerOverrides, TestSetMeshShape) {
  
  ListOption<int64_t> meshShape;
  meshShape.push_back(1);
  meshShape.push_back(2);

  optimizerOverridesHandler->setMeshShape(meshShape);
  ASSERT_EQ(optimizerOverridesHandler->getMeshShape()[0], meshShape[0]);
  ASSERT_EQ(optimizerOverridesHandler->getMeshShape()[1], meshShape[1]);

}

// Test the toString method
TEST_F(TestOptimizerOverrides, TestToString) {

  std::string options = "memreconfig-enabled=true memory-layout-analysis-enabled=true insert-memreconfig=add_0_1_2=0 override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32";

  optimizerOverridesHandler->setMemoryLayoutAnalysis(true);
  optimizerOverridesHandler->setMemoryConfig(true);
  optimizerOverridesHandler->addInputLayoutOverride("add_0_1_2", { 0 });
  optimizerOverridesHandler->addOutputLayoutOverride("add_1_2", { 1, 1 }, tt::MemorySpace::DeviceDRAM, tt::TensorMemoryLayout::Interleaved, tt::ttnn::Layout::Row_Major, tt::DataType::Float32);

  ASSERT_EQ(optimizerOverridesHandler->toString(), options);

}
