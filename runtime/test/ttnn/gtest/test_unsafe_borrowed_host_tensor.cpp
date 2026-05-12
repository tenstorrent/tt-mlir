// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

class UnsafeBorrowedHostTensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
  }
};

} // namespace

TEST_F(UnsafeBorrowedHostTensorTest, AliasesOwnedBuffer) {
  std::vector<uint32_t> shape = {2, 2};
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::Float32;
  uint32_t itemSize = sizeof(float);

  std::vector<float> testData = {1.0f, 2.0f, 3.0f, 4.0f};

  tt::runtime::Tensor owned = tt::runtime::createOwnedHostTensor(
      testData.data(), shape, stride, itemSize, dataType);

  tt::runtime::Tensor borrowed =
      tt::runtime::createUnsafeBorrowedHostTensor(owned);

  const ::ttnn::Tensor &ownedTtnn =
      tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(owned);
  const ::ttnn::Tensor &borrowedTtnn =
      tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(borrowed);

  EXPECT_EQ(tt::runtime::ttnn::utils::getRawHostDataPtr(ownedTtnn),
            tt::runtime::ttnn::utils::getRawHostDataPtr(borrowedTtnn));
  EXPECT_EQ(tt::runtime::getTensorShape(owned),
            tt::runtime::getTensorShape(borrowed));
  EXPECT_EQ(tt::runtime::getTensorDataType(owned),
            tt::runtime::getTensorDataType(borrowed));
}
