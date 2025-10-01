// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/test/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
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
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
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
  std::vector<uint32_t> shape = {3, 3};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::Float32;
  uint32_t itemSize = sizeof(float);

  // Create test data with specific float values
  std::vector<float> testData = {1.0f,  2.5f,      3.14159f, -1.0f, 0.0f,
                                 42.0f, -3.14159f, 1e6f,     -1e-6f};

  tt::runtime::Tensor originalTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  std::string tempFilePath = "/tmp/test_float32_tensor.bin";
  addTempFile(tempFilePath);

  ASSERT_NO_THROW(tt::runtime::dumpTensor(originalTensor, tempFilePath));

  tt::runtime::Tensor loadedTensor;
  ASSERT_NO_THROW(loadedTensor = tt::runtime::loadTensor(tempFilePath));

  // Verify tensor properties match
  EXPECT_EQ(tt::runtime::getTensorShape(originalTensor),
            tt::runtime::getTensorShape(loadedTensor))
      << "Shape mismatch for device tensor";

  EXPECT_EQ(tt::runtime::getTensorDataType(originalTensor),
            tt::runtime::getTensorDataType(loadedTensor))
      << "Data type mismatch for device tensor";

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

TEST_F(TensorSerializationTest, TestDumpAndLoadTensorWithDevice) {
  std::vector<uint32_t> shape = {32, 32};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::BFloat16;
  uint32_t itemSize = sizeof(uint16_t);

  std::vector<uint16_t> testData;
  for (size_t i = 0; i < shape[0] * shape[1]; ++i) {
    testData.push_back(static_cast<uint16_t>(i % 256));
  }

  tt::runtime::Tensor hostTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  tt::runtime::MeshDeviceOptions meshOptions;
  meshOptions.meshShape = {1, 1};
  tt::runtime::Device device = tt::runtime::openMeshDevice(meshOptions);

  tt::runtime::Layout deviceLayout =
      tt::runtime::test::ttnn::getDramInterleavedTileLayout(dataType);

  tt::runtime::Tensor deviceTensor;
  ASSERT_NO_THROW(deviceTensor = tt::runtime::toLayout(hostTensor, device,
                                                       deviceLayout, true));

  std::string tempFilePath = "/tmp/test_tensor_device_serialization.bin";
  addTempFile(tempFilePath);

  ASSERT_NO_THROW(tt::runtime::dumpTensor(deviceTensor, tempFilePath));

  tt::runtime::Tensor loadedDeviceTensor;
  ASSERT_NO_THROW(loadedDeviceTensor =
                      tt::runtime::loadTensor(tempFilePath, device));

  // Verify that the loaded tensor is actually on device
  auto &loadedTTNNTensor =
      tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
          loadedDeviceTensor);
  EXPECT_TRUE(
      tt::runtime::ttnn::utils::isOnDevice(loadedTTNNTensor.storage_type()))
      << "Loaded tensor should be on device but is not";

  std::vector<tt::runtime::Tensor> originalHostTensors;
  std::vector<tt::runtime::Tensor> loadedHostTensors;

  ASSERT_NO_THROW(originalHostTensors =
                      tt::runtime::toHost(deviceTensor, true, true));
  ASSERT_NO_THROW(loadedHostTensors =
                      tt::runtime::toHost(loadedDeviceTensor, true, true));

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

  // Clean up device
  tt::runtime::closeMeshDevice(device);
}
