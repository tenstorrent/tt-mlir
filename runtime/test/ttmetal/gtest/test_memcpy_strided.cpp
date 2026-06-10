// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

class TTMetalMemcpyStridedTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTMetal);
  }

  static tt::runtime::Tensor makeBorrowed(float *data,
                                          const std::vector<uint32_t> &shape,
                                          const std::vector<int64_t> &stride,
                                          uint64_t physicalVolume) {
    tt::runtime::TensorDesc desc(shape, tt::target::DataType::Float32, stride,
                                 physicalVolume);
    return tt::runtime::ttmetal::createBorrowedHostTensor(data, desc);
  }
};

} // namespace

TEST_F(TTMetalMemcpyStridedTest, CopiesPaddedRankOneSource) {
  std::vector<float> padded = {
      0.0f, 99.0f, 99.0f, 1.0f, 99.0f, 99.0f,
      2.0f, 99.0f, 99.0f, 3.0f, 99.0f, 99.0f,
  };
  std::vector<float> dense(4, -1.0f);

  tt::runtime::Tensor src = makeBorrowed(padded.data(), {4}, {3}, 12);
  tt::runtime::Tensor dst = makeBorrowed(dense.data(), {4}, {1}, 4);

  tt::runtime::memcpy(dst, src);

  std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f};
  EXPECT_EQ(dense, expected);
}

TEST_F(TTMetalMemcpyStridedTest, CopiesPaddedRankTwoSource) {
  std::vector<float> padded(32, 99.0f);
  padded[0] = 0.0f;
  padded[2] = 1.0f;
  padded[4] = 2.0f;
  padded[6] = 3.0f;
  padded[16] = 4.0f;
  padded[18] = 5.0f;
  padded[20] = 6.0f;
  padded[22] = 7.0f;
  std::vector<float> dense(8, -1.0f);

  tt::runtime::Tensor src = makeBorrowed(padded.data(), {2, 4}, {16, 2}, 32);
  tt::runtime::Tensor dst = makeBorrowed(dense.data(), {2, 4}, {4, 1}, 8);

  tt::runtime::memcpy(dst, src);

  std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f, 7.0f};
  EXPECT_EQ(dense, expected);
}

TEST_F(TTMetalMemcpyStridedTest, CopiesPaddedScalarSource) {
  std::vector<float> padded = {42.0f, 99.0f, 99.0f, 99.0f};
  std::vector<float> dense(1, -1.0f);

  tt::runtime::Tensor src = makeBorrowed(padded.data(), {}, {}, 4);
  tt::runtime::Tensor dst = makeBorrowed(dense.data(), {}, {}, 1);

  tt::runtime::memcpy(dst, src);

  EXPECT_EQ(dense[0], 42.0f);
}
