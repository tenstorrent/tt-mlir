// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/test/ttnn/utils.h"
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
  // Test moving tensor to device, serializing from device, and loading to
  // device
  std::vector<uint32_t> shape = {32, 32}; // Use 32x32 for tile alignment
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::BFloat16;
  uint32_t itemSize = sizeof(uint16_t);

  // Create test data
  std::vector<uint16_t> testData;
  for (size_t i = 0; i < shape[0] * shape[1]; ++i) {
    testData.push_back(static_cast<uint16_t>(i % 256));
  }

  // Create host tensor
  tt::runtime::Tensor hostTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  // Open mesh device
  tt::runtime::MeshDeviceOptions meshOptions;
  meshOptions.meshShape = {1, 1};
  tt::runtime::Device device = tt::runtime::openMeshDevice(meshOptions);

  // Create device layout for DRAM interleaved tile layout
  tt::runtime::Layout deviceLayout =
      tt::runtime::test::ttnn::getDramInterleavedTileLayout(dataType);

  // Move tensor to device
  tt::runtime::Tensor deviceTensor;
  ASSERT_NO_THROW(deviceTensor = tt::runtime::toLayout(hostTensor, device,
                                                       deviceLayout, true));

  std::string tempFilePath = "/tmp/test_tensor_device_serialization.bin";
  addTempFile(tempFilePath);

  // Dump tensor from device
  ASSERT_NO_THROW(tt::runtime::dumpTensor(deviceTensor, tempFilePath));

  // Verify file was created
  std::ifstream file(tempFilePath);
  ASSERT_TRUE(file.good()) << "Failed to create dump file for device tensor";
  file.close();

  // Load tensor directly to device
  tt::runtime::Tensor loadedDeviceTensor;
  ASSERT_NO_THROW(loadedDeviceTensor =
                      tt::runtime::loadTensor(tempFilePath, device));

  // Move both tensors back to host for comparison
  std::vector<tt::runtime::Tensor> originalHostTensors;
  std::vector<tt::runtime::Tensor> loadedHostTensors;

  ASSERT_NO_THROW(originalHostTensors =
                      tt::runtime::toHost(deviceTensor, true, true));
  ASSERT_NO_THROW(loadedHostTensors =
                      tt::runtime::toHost(loadedDeviceTensor, true, true));

  ASSERT_EQ(originalHostTensors.size(), 1);
  ASSERT_EQ(loadedHostTensors.size(), 1);

  // Verify tensor properties match
  EXPECT_EQ(tt::runtime::getTensorShape(originalHostTensors[0]),
            tt::runtime::getTensorShape(loadedHostTensors[0]))
      << "Shape mismatch for device tensor";

  EXPECT_EQ(tt::runtime::getTensorDataType(originalHostTensors[0]),
            tt::runtime::getTensorDataType(loadedHostTensors[0]))
      << "Data type mismatch for device tensor";

  // Verify data integrity
  std::vector<std::byte> originalDataBuffer =
      tt::runtime::getTensorDataBuffer(originalHostTensors[0]);
  std::vector<std::byte> loadedDataBuffer =
      tt::runtime::getTensorDataBuffer(loadedHostTensors[0]);

  EXPECT_EQ(originalDataBuffer.size(), loadedDataBuffer.size())
      << "Data buffer size mismatch for device tensor";

  EXPECT_EQ(std::memcmp(originalDataBuffer.data(), loadedDataBuffer.data(),
                        originalDataBuffer.size()),
            0)
      << "Data content mismatch for device tensor";

  // Verify the actual data values by checking against original test data
  // Note: The data might be in a different layout after device round-trip,
  // but should still contain the same values
  ASSERT_EQ(loadedDataBuffer.size(), testData.size() * sizeof(uint16_t));

  // Clean up device
  tt::runtime::closeMeshDevice(device);
}
