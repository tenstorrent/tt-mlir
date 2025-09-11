// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

class TensorSerializationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up the runtime for TTNN
    tt::runtime::setCurrentRuntime(tt::runtime::DeviceRuntime::TTNN);
  }

  void TearDown() override {
    // Clean up any temporary files that might have been left behind
    for (const std::string &file : tempFiles) {
      std::remove(file.c_str());
    }
  }

  void addTempFile(const std::string &filePath) {
    tempFiles.push_back(filePath);
  }

private:
  std::vector<std::string> tempFiles;
};

TEST_F(TensorSerializationTest, TestDumpAndLoadTensorBasic) {
  // Test data types and shapes
  struct TestCase {
    std::vector<uint32_t> shape;
    tt::target::DataType dataType;
    uint32_t itemSize;
    std::string testName;
  };

  std::vector<TestCase> testCases = {
      {{4, 4}, tt::target::DataType::Float32, sizeof(float), "Float32_4x4"},
      {{2, 3, 4},
       tt::target::DataType::Float16,
       sizeof(uint16_t),
       "Float16_2x3x4"},
      {{8}, tt::target::DataType::BFloat16, sizeof(uint16_t), "BFloat16_8"},
      {{2, 2, 2, 2},
       tt::target::DataType::UInt32,
       sizeof(uint32_t),
       "UInt32_2x2x2x2"}};

  for (const auto &testCase : testCases) {
    // Calculate stride
    std::vector<uint32_t> stride =
        tt::runtime::utils::calculateStride(testCase.shape);

    // Calculate total number of elements
    uint32_t totalElements = 1;
    for (uint32_t dim : testCase.shape) {
      totalElements *= dim;
    }

    // Create test data based on data type
    std::vector<std::byte> originalData(totalElements * testCase.itemSize);

    // Fill with predictable test pattern
    for (size_t i = 0; i < originalData.size(); ++i) {
      originalData[i] = static_cast<std::byte>((i + 1) % 256);
    }

    // Create tensor with owned data
    tt::runtime::Tensor originalTensor = tt::runtime::createOwnedHostTensor(
        originalData.data(), testCase.shape, stride, testCase.itemSize,
        testCase.dataType);

    // Create temporary file path
    std::string tempFilePath = "/tmp/test_tensor_" + testCase.testName + ".bin";
    addTempFile(tempFilePath);

    // Test dump tensor
    ASSERT_NO_THROW(tt::runtime::dumpTensor(originalTensor, tempFilePath));

    // Verify file was created
    std::ifstream file(tempFilePath);
    ASSERT_TRUE(file.good())
        << "Failed to create dump file for " << testCase.testName;
    file.close();

    // Test load tensor
    tt::runtime::Tensor loadedTensor;
    ASSERT_NO_THROW(loadedTensor = tt::runtime::loadTensor(tempFilePath));

    // Verify tensor properties match
    EXPECT_EQ(tt::runtime::getTensorShape(originalTensor),
              tt::runtime::getTensorShape(loadedTensor))
        << "Shape mismatch for " << testCase.testName;

    EXPECT_EQ(tt::runtime::getTensorStride(originalTensor),
              tt::runtime::getTensorStride(loadedTensor))
        << "Stride mismatch for " << testCase.testName;

    EXPECT_EQ(tt::runtime::getTensorElementSize(originalTensor),
              tt::runtime::getTensorElementSize(loadedTensor))
        << "Element size mismatch for " << testCase.testName;

    EXPECT_EQ(tt::runtime::getTensorDataType(originalTensor),
              tt::runtime::getTensorDataType(loadedTensor))
        << "Data type mismatch for " << testCase.testName;

    // Verify tensor data is identical
    std::vector<std::byte> originalDataBuffer =
        tt::runtime::getTensorDataBuffer(originalTensor);
    std::vector<std::byte> loadedDataBuffer =
        tt::runtime::getTensorDataBuffer(loadedTensor);

    EXPECT_EQ(originalDataBuffer.size(), loadedDataBuffer.size())
        << "Data buffer size mismatch for " << testCase.testName;

    EXPECT_EQ(std::memcmp(originalDataBuffer.data(), loadedDataBuffer.data(),
                          originalDataBuffer.size()),
              0)
        << "Data content mismatch for " << testCase.testName;
  }
}

TEST_F(TensorSerializationTest, TestDumpAndLoadFloat32Tensor) {
  // Simple test with float data
  std::vector<uint32_t> shape = {3, 3};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::Float32;
  uint32_t itemSize = sizeof(float);

  // Create test data with specific float values
  std::vector<float> testData = {1.0f,  2.5f,      3.14159f, -1.0f, 0.0f,
                                 42.0f, -3.14159f, 1e6f,     -1e-6f};

  // Create tensor
  tt::runtime::Tensor originalTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  std::string tempFilePath = "/tmp/test_float32_tensor.bin";
  addTempFile(tempFilePath);

  // Dump tensor
  ASSERT_NO_THROW(tt::runtime::dumpTensor(originalTensor, tempFilePath));

  // Load tensor
  tt::runtime::Tensor loadedTensor;
  ASSERT_NO_THROW(loadedTensor = tt::runtime::loadTensor(tempFilePath));

  // Verify data integrity
  std::vector<std::byte> originalDataBuffer =
      tt::runtime::getTensorDataBuffer(originalTensor);
  std::vector<std::byte> loadedDataBuffer =
      tt::runtime::getTensorDataBuffer(loadedTensor);

  EXPECT_EQ(originalDataBuffer.size(), loadedDataBuffer.size());
  EXPECT_EQ(std::memcmp(originalDataBuffer.data(), loadedDataBuffer.data(),
                        originalDataBuffer.size()),
            0);

  // Also verify by interpreting the bytes back as floats
  ASSERT_EQ(loadedDataBuffer.size(), testData.size() * sizeof(float));
  const float *loadedFloats =
      reinterpret_cast<const float *>(loadedDataBuffer.data());

  for (size_t i = 0; i < testData.size(); ++i) {
    EXPECT_FLOAT_EQ(testData[i], loadedFloats[i])
        << "Float value mismatch at index " << i;
  }
}

TEST_F(TensorSerializationTest, TestDumpAndLoadTensorWithDevice) {
  // Test with device parameter
  std::vector<uint32_t> shape = {2, 4};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::UInt32;
  uint32_t itemSize = sizeof(uint32_t);

  // Create test data
  std::vector<uint32_t> testData = {1, 2, 3, 4, 5, 6, 7, 8};

  // Create tensor
  tt::runtime::Tensor originalTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  std::string tempFilePath = "/tmp/test_tensor_with_device.bin";
  addTempFile(tempFilePath);

  // Dump tensor
  ASSERT_NO_THROW(tt::runtime::dumpTensor(originalTensor, tempFilePath));

  // Load tensor with explicit device parameter (as nullopt)
  tt::runtime::Tensor loadedTensor;
  ASSERT_NO_THROW(loadedTensor =
                      tt::runtime::loadTensor(tempFilePath, std::nullopt));

  // Verify data integrity
  std::vector<std::byte> originalDataBuffer =
      tt::runtime::getTensorDataBuffer(originalTensor);
  std::vector<std::byte> loadedDataBuffer =
      tt::runtime::getTensorDataBuffer(loadedTensor);

  EXPECT_EQ(originalDataBuffer.size(), loadedDataBuffer.size());
  EXPECT_EQ(std::memcmp(originalDataBuffer.data(), loadedDataBuffer.data(),
                        originalDataBuffer.size()),
            0);

  // Verify the actual uint32 values
  ASSERT_EQ(loadedDataBuffer.size(), testData.size() * sizeof(uint32_t));
  const uint32_t *loadedInts =
      reinterpret_cast<const uint32_t *>(loadedDataBuffer.data());

  for (size_t i = 0; i < testData.size(); ++i) {
    EXPECT_EQ(testData[i], loadedInts[i])
        << "UInt32 value mismatch at index " << i;
  }
}

TEST_F(TensorSerializationTest, TestLoadNonExistentFile) {
  // Test loading a non-existent file
  std::string nonExistentFile = "/tmp/non_existent_tensor_file_12345.bin";

  // This should throw an exception or handle gracefully
  EXPECT_THROW(tt::runtime::loadTensor(nonExistentFile), std::runtime_error);
}

TEST_F(TensorSerializationTest, TestDumpToInvalidPath) {
  // Create a simple tensor
  std::vector<uint32_t> shape = {2, 2};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  tt::runtime::Tensor tensor = tt::runtime::createOwnedHostTensor(
      data.data(), shape, stride, sizeof(float), tt::target::DataType::Float32);

  // Try to dump to an invalid path (non-existent directory)
  std::string invalidPath = "/invalid/path/that/does/not/exist/tensor.bin";

  // This should throw an exception or handle gracefully
  EXPECT_THROW(tt::runtime::dumpTensor(tensor, invalidPath),
               std::runtime_error);
}

TEST_F(TensorSerializationTest, TestEmptyTensor) {
  // Test with an empty tensor (shape with zero elements)
  std::vector<uint32_t> shape = {0};
  std::vector<uint32_t> stride = {1};
  tt::target::DataType dataType = tt::target::DataType::Float32;
  uint32_t itemSize = sizeof(float);

  // Create empty data vector
  std::vector<float> emptyData;

  // Create tensor with empty data
  tt::runtime::Tensor originalTensor = tt::runtime::createOwnedHostTensor(
      emptyData.data(), shape, stride, itemSize, dataType);

  std::string tempFilePath = "/tmp/test_empty_tensor.bin";
  addTempFile(tempFilePath);

  // Test dump and load
  ASSERT_NO_THROW(tt::runtime::dumpTensor(originalTensor, tempFilePath));

  tt::runtime::Tensor loadedTensor;
  ASSERT_NO_THROW(loadedTensor = tt::runtime::loadTensor(tempFilePath));

  // Verify properties
  EXPECT_EQ(tt::runtime::getTensorShape(originalTensor),
            tt::runtime::getTensorShape(loadedTensor));
  EXPECT_EQ(tt::runtime::getTensorDataType(originalTensor),
            tt::runtime::getTensorDataType(loadedTensor));

  // Verify data buffers
  std::vector<std::byte> originalDataBuffer =
      tt::runtime::getTensorDataBuffer(originalTensor);
  std::vector<std::byte> loadedDataBuffer =
      tt::runtime::getTensorDataBuffer(loadedTensor);
  EXPECT_EQ(originalDataBuffer.size(), loadedDataBuffer.size());
}

TEST_F(TensorSerializationTest, TestLargeTensor) {
  // Test with a larger tensor to ensure it handles bigger data correctly
  std::vector<uint32_t> shape = {100, 100};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::Float32;
  uint32_t itemSize = sizeof(float);

  // Create large test data with a pattern
  std::vector<float> testData;
  testData.reserve(10000);

  for (int i = 0; i < 10000; ++i) {
    testData.push_back(static_cast<float>(i % 1000) + 0.5f);
  }

  // Create tensor
  tt::runtime::Tensor originalTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  std::string tempFilePath = "/tmp/test_large_tensor.bin";
  addTempFile(tempFilePath);

  // Test dump and load
  ASSERT_NO_THROW(tt::runtime::dumpTensor(originalTensor, tempFilePath));

  tt::runtime::Tensor loadedTensor;
  ASSERT_NO_THROW(loadedTensor = tt::runtime::loadTensor(tempFilePath));

  // Verify tensor properties
  EXPECT_EQ(tt::runtime::getTensorShape(originalTensor),
            tt::runtime::getTensorShape(loadedTensor));
  EXPECT_EQ(tt::runtime::getTensorDataType(originalTensor),
            tt::runtime::getTensorDataType(loadedTensor));

  // Verify data integrity
  std::vector<std::byte> originalDataBuffer =
      tt::runtime::getTensorDataBuffer(originalTensor);
  std::vector<std::byte> loadedDataBuffer =
      tt::runtime::getTensorDataBuffer(loadedTensor);

  EXPECT_EQ(originalDataBuffer.size(), loadedDataBuffer.size());
  EXPECT_EQ(std::memcmp(originalDataBuffer.data(), loadedDataBuffer.data(),
                        originalDataBuffer.size()),
            0);
}
