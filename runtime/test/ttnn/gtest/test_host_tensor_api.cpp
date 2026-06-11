// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

class HostTensorTest : public ::testing::Test {
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

  static tt::runtime::Tensor makeBorrowed(std::vector<float> &data,
                                          const std::vector<uint32_t> &shape,
                                          const std::vector<int64_t> &stride) {
    return tt::runtime::createBorrowedHostTensor(data.data(), shape, stride,
                                                 sizeof(float),
                                                 tt::target::DataType::Float32);
  }
};

// Tests for strided (non-contiguous) input buffers.
class OwnedHostTensorStridedTest : public HostTensorTest {};

// Tests for 0-dim (scalar) tensors with empty shape/stride across the owned
// tensor APIs.
class OwnedHostTensorScalarTest : public HostTensorTest {};

// Tests for the borrowed (zero-copy) tensor APIs, single- and multi-device.
class BorrowedHostTensorTest : public HostTensorTest {};

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

// An empty shape/stride creates a 0-dim (scalar) tensor holding exactly one
// element.
TEST_F(OwnedHostTensorScalarTest, CreatesScalarTensor) {
  float value = 42.0f;

  tt::runtime::Tensor tensor = tt::runtime::createOwnedHostTensor(
      &value, /*shape=*/{}, /*stride=*/{}, sizeof(float),
      tt::target::DataType::Float32);

  EXPECT_TRUE(tt::runtime::getTensorShape(tensor).empty());
  EXPECT_EQ(tt::runtime::getTensorVolume(tensor), 1u);
  EXPECT_EQ(readFloats(tensor), std::vector<float>{42.0f});
}

// createMultiDeviceHostTensor accepts 0-dim (scalar) shards the same way
// createOwnedHostTensor does.
TEST_F(OwnedHostTensorScalarTest, CreatesMultiDeviceScalarTensor) {
  float shard0 = 1.0f;
  float shard1 = 2.0f;
  std::vector<const void *> shards = {&shard0, &shard1};

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceHostTensor(
      shards, /*shape=*/{}, /*stride=*/{}, sizeof(float),
      tt::target::DataType::Float32, /*strategy=*/{}, /*meshShape=*/{1, 2});

  EXPECT_TRUE(tt::runtime::getTensorShape(tensor).empty());

  std::vector<tt::runtime::Tensor> hostShards =
      tt::runtime::getDeviceTensors(tensor);
  ASSERT_EQ(hostShards.size(), 2u);
  EXPECT_EQ(readFloats(hostShards[0]), std::vector<float>{1.0f});
  EXPECT_EQ(readFloats(hostShards[1]), std::vector<float>{2.0f});
}

// The from-shards overload must also accept 0-dim (scalar) shards.
TEST_F(OwnedHostTensorScalarTest, CreatesMultiDeviceScalarTensorFromShards) {
  float shard0 = 1.0f;
  float shard1 = 2.0f;
  std::vector<tt::runtime::Tensor> shards = {
      tt::runtime::createOwnedHostTensor(&shard0, /*shape=*/{}, /*stride=*/{},
                                         sizeof(float),
                                         tt::target::DataType::Float32),
      tt::runtime::createOwnedHostTensor(&shard1, /*shape=*/{}, /*stride=*/{},
                                         sizeof(float),
                                         tt::target::DataType::Float32)};

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceHostTensor(
      shards, /*strategy=*/{}, /*meshShape=*/{1, 2});

  EXPECT_TRUE(tt::runtime::getTensorShape(tensor).empty());

  std::vector<tt::runtime::Tensor> hostShards =
      tt::runtime::getDeviceTensors(tensor);
  ASSERT_EQ(hostShards.size(), 2u);
  EXPECT_EQ(readFloats(hostShards[0]), std::vector<float>{1.0f});
  EXPECT_EQ(readFloats(hostShards[1]), std::vector<float>{2.0f});
}

// A borrowed tensor aliases the caller's buffer instead of copying it, so
// writes to the buffer after creation must be visible through the tensor.
TEST_F(BorrowedHostTensorTest, AliasesContiguousBuffer) {
  std::vector<float> data = {1, 2, 3, 4};
  std::vector<uint32_t> shape = {2, 2};
  std::vector<int64_t> stride = {2, 1};

  tt::runtime::Tensor tensor = makeBorrowed(data, shape, stride);

  EXPECT_EQ(readFloats(tensor), data);

  data[0] = 99.0f;
  EXPECT_EQ(readFloats(tensor), data);
}

// We cannot borrow a non-contiguous buffer.
TEST_F(BorrowedHostTensorTest, RejectsNonContiguousBuffer) {
  std::vector<float> parent = {0, 1, 2, 3, 4, 5};
  std::vector<uint32_t> shape = {3, 2};
  std::vector<int64_t> stride = {1, 3}; // transposed view of a 2x3 parent

  EXPECT_THROW(makeBorrowed(parent, shape, stride), std::runtime_error);
}

// An empty shape/stride creates a borrowed 0-dim (scalar) tensor, same as the
// owned variant.
TEST_F(BorrowedHostTensorTest, CreatesScalarTensor) {
  float value = 42.0f;

  tt::runtime::Tensor tensor = tt::runtime::createBorrowedHostTensor(
      &value, /*shape=*/{}, /*stride=*/{}, sizeof(float),
      tt::target::DataType::Float32);

  EXPECT_TRUE(tt::runtime::getTensorShape(tensor).empty());
  EXPECT_EQ(tt::runtime::getTensorVolume(tensor), 1u);
  EXPECT_EQ(readFloats(tensor), std::vector<float>{42.0f});
}

// A multi-device borrowed tensor aliases every shard's buffer, so writes to
// the buffers after creation must be visible through the shards.
TEST_F(BorrowedHostTensorTest, CreatesMultiDeviceBorrowedTensor) {
  std::vector<float> shard0 = {1, 2, 3, 4};
  std::vector<float> shard1 = {5, 6, 7, 8};
  std::vector<void *> shards = {shard0.data(), shard1.data()};
  std::vector<uint32_t> shape = {2, 2};
  std::vector<int64_t> stride = {2, 1};

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceBorrowedHostTensor(
      shards, shape, stride, sizeof(float), tt::target::DataType::Float32,
      /*strategy=*/{}, /*meshShape=*/{1, 2});

  std::vector<tt::runtime::Tensor> hostShards =
      tt::runtime::getDeviceTensors(tensor);
  ASSERT_EQ(hostShards.size(), 2u);
  EXPECT_EQ(readFloats(hostShards[0]), shard0);
  EXPECT_EQ(readFloats(hostShards[1]), shard1);

  shard0[0] = 99.0f;
  shard1[3] = -1.0f;
  EXPECT_EQ(readFloats(hostShards[0]), shard0);
  EXPECT_EQ(readFloats(hostShards[1]), shard1);
}

// createMultiDeviceBorrowedHostTensor must accept 0-dim (scalar) shards the
// same way the single-device borrowed and multi-device owned variants do.
TEST_F(BorrowedHostTensorTest, CreatesMultiDeviceBorrowedScalarTensor) {
  float shard0 = 1.0f;
  float shard1 = 2.0f;
  std::vector<void *> shards = {&shard0, &shard1};

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceBorrowedHostTensor(
      shards, /*shape=*/{}, /*stride=*/{}, sizeof(float),
      tt::target::DataType::Float32, /*strategy=*/{}, /*meshShape=*/{1, 2});

  EXPECT_TRUE(tt::runtime::getTensorShape(tensor).empty());

  std::vector<tt::runtime::Tensor> hostShards =
      tt::runtime::getDeviceTensors(tensor);
  ASSERT_EQ(hostShards.size(), 2u);
  EXPECT_EQ(readFloats(hostShards[0]), std::vector<float>{1.0f});
  EXPECT_EQ(readFloats(hostShards[1]), std::vector<float>{2.0f});
}
