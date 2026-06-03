// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

class OwnedHostTensorStridedTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
  }

  // Reads back the dense, row-major float contents of a host tensor.
  static std::vector<float> readFloats(tt::runtime::Tensor tensor) {
    std::vector<std::byte> bytes = tt::runtime::getTensorDataBuffer(tensor);
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  }

  static tt::runtime::Tensor makeOwned(const std::vector<float> &data,
                                       const std::vector<uint32_t> &shape,
                                       const std::vector<int64_t> &stride) {
    return tt::runtime::createOwnedHostTensor(data.data(), shape, stride,
                                              sizeof(float),
                                              tt::target::DataType::Float32);
  }
};

} // namespace

// A contiguous buffer must be copied through unchanged (no spurious gather).
TEST_F(OwnedHostTensorStridedTest, ContiguousIsUnchanged) {
  std::vector<float> data = {0, 1, 2, 3, 4, 5}; // 2x3 row-major
  std::vector<uint32_t> shape = {2, 3};
  std::vector<int64_t> stride = {3, 1}; // dense row-major strides

  tt::runtime::Tensor tensor = makeOwned(data, shape, stride);

  EXPECT_EQ(readFloats(tensor), data);
}

// A transposed (non-contiguous) view must be gathered into dense row-major
// order. parent 2x3 = [0..5]; view = parent.T -> shape [3,2], strides [1,3].
TEST_F(OwnedHostTensorStridedTest, GathersTransposedView) {
  std::vector<float> parent = {0, 1, 2, 3, 4, 5};
  std::vector<uint32_t> shape = {3, 2};
  std::vector<int64_t> stride = {1, 3};

  tt::runtime::Tensor tensor = makeOwned(parent, shape, stride);

  std::vector<float> expected = {0, 3, 1, 4, 2, 5};
  EXPECT_EQ(readFloats(tensor), expected);
}

// A sliced view with gaps must skip the unused elements. parent 3x4 = [0..11];
// view = parent[:, 0:2] -> shape [3,2], strides [4,1] (the row stride spans the
// full parent row of 4, so columns 2 and 3 are skipped).
TEST_F(OwnedHostTensorStridedTest, GathersSlicedView) {
  std::vector<float> parent = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<uint32_t> shape = {3, 2};
  std::vector<int64_t> stride = {4, 1};

  tt::runtime::Tensor tensor = makeOwned(parent, shape, stride);

  std::vector<float> expected = {0, 1, 4, 5, 8, 9};
  EXPECT_EQ(readFloats(tensor), expected);
}

// A reversed (negative-stride) view must walk backward from the data pointer,
// which points at the first logical element. parent = [0,1,2,3];
// view = parent[::-1] -> shape [4], stride [-1], data pointer at parent[3].
TEST_F(OwnedHostTensorStridedTest, GathersReversedView) {
  std::vector<float> parent = {0, 1, 2, 3};
  std::vector<uint32_t> shape = {4};
  std::vector<int64_t> stride = {-1};

  tt::runtime::Tensor tensor = tt::runtime::createOwnedHostTensor(
      parent.data() + 3, shape, stride, sizeof(float),
      tt::target::DataType::Float32);

  std::vector<float> expected = {3, 2, 1, 0};
  EXPECT_EQ(readFloats(tensor), expected);
}
