// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/test/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

class LayoutConverterTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
  }
};

// Regression for PJRT ensure_layout / RowMajorLayoutPropagation: device-side
// ROW_MAJOR tensors that only need a dtype change used to LOG_FATAL in
// handleDeviceInputNoLayoutTypecast. ttnn::typecast supports ROW_MAJOR on
// device, so toLayout must succeed without a host round-trip.
TEST_F(LayoutConverterTest, DeviceToDeviceRowMajorTypecast) {
  std::vector<uint32_t> shape = {32, 32};
  std::vector<int64_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType srcDataType = tt::target::DataType::Float32;
  tt::target::DataType dstDataType = tt::target::DataType::BFloat16;
  uint32_t itemSize = sizeof(float);

  std::vector<float> testData(shape[0] * shape[1]);
  for (size_t i = 0; i < testData.size(); ++i) {
    testData[i] = static_cast<float>(i);
  }

  tt::runtime::Tensor hostTensor = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, srcDataType);

  tt::runtime::MeshDeviceOptions meshOptions;
  meshOptions.meshShape = {1, 1};
  tt::runtime::Device device = tt::runtime::openMeshDevice(meshOptions);

  tt::runtime::Layout srcLayout =
      tt::runtime::test::ttnn::getDramInterleavedRowMajorLayout(srcDataType);
  tt::runtime::Layout dstLayout =
      tt::runtime::test::ttnn::getDramInterleavedRowMajorLayout(dstDataType);

  tt::runtime::Tensor deviceTensor;
  ASSERT_NO_THROW(deviceTensor = tt::runtime::toLayout(hostTensor, device,
                                                       srcLayout, true));
  ASSERT_TRUE(tt::runtime::hasLayout(deviceTensor, srcLayout));

  tt::runtime::Tensor typedDeviceTensor;
  ASSERT_NO_THROW(typedDeviceTensor = tt::runtime::toLayout(
                      deviceTensor, device, dstLayout, true))
      << "Device-to-device ROW_MAJOR typecast must not fatal";

  EXPECT_TRUE(tt::runtime::hasLayout(typedDeviceTensor, dstLayout));
  EXPECT_EQ(tt::runtime::getTensorDataType(typedDeviceTensor), dstDataType);

  auto &ttnnTensor = tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
      typedDeviceTensor);
  EXPECT_TRUE(tt::runtime::ttnn::utils::isOnDevice(ttnnTensor.storage_type()))
      << "Typecast result should remain on device";

  tt::runtime::closeMeshDevice(device);
}
