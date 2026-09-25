// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test: isTensorAllocated for TTMetal tensors.
//
// Background: builder_runtime.py calls is_allocated() before to_host() to
// skip unallocated const-eval output tensors.
//
// TTMetal allocation status depends on the MetalTensor variant:
//  - TensorDesc with non-null data: allocated (HostAllocCommand output)
//  - TensorDesc with null data: unallocated (const-eval placeholder)
//  - HostBuffer: allocated (MeshShardCommand shard_to_full output)
//  - DistributedHostBuffer: allocated (MeshShardCommand full_to_shard output)
//  - MeshBuffer: unallocated (to_host not yet implemented; skip gracefully)

#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

using tt::runtime::DeviceRuntime;
using tt::runtime::Tensor;
using tt::runtime::TensorDesc;
using tt::runtime::ttmetal::DistributedHostBuffer;
using tt::runtime::ttmetal::HostBuffer;
using tt::runtime::ttmetal::MeshBuffer;
using tt::runtime::ttmetal::MetalTensor;

class IsTensorAllocatedTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(DeviceRuntime::TTMetal);
  }
};

// TensorDesc + null data: unallocated const-eval placeholder.
TEST_F(IsTensorAllocatedTest, ReturnsFalseForTensorDescNullData) {
  TensorDesc desc({1, 4}, tt::target::DataType::Float32);
  auto handle = std::make_shared<MetalTensor>(desc);
  Tensor tensor(std::static_pointer_cast<void>(handle), nullptr,
                DeviceRuntime::TTMetal);

  EXPECT_FALSE(tt::runtime::isTensorAllocated(tensor));
}

// TensorDesc + non-null data: allocated HostAllocCommand output.
TEST_F(IsTensorAllocatedTest, ReturnsTrueForTensorDescNonNullData) {
  TensorDesc desc({1, 4}, tt::target::DataType::Float32);
  auto handle = std::make_shared<MetalTensor>(desc);
  auto data = std::shared_ptr<void>(std::calloc(16, 1), std::free);
  Tensor tensor(std::static_pointer_cast<void>(handle), data,
                DeviceRuntime::TTMetal);

  EXPECT_TRUE(tt::runtime::isTensorAllocated(tensor));
}

// HostBuffer (null inner pointer is fine): allocated MeshShard shard_to_full.
TEST_F(IsTensorAllocatedTest, ReturnsTrueForHostBuffer) {
  HostBuffer nullBuffer;
  auto handle = std::make_shared<MetalTensor>(nullBuffer);
  Tensor tensor(std::static_pointer_cast<void>(handle), nullptr,
                DeviceRuntime::TTMetal);

  EXPECT_TRUE(tt::runtime::isTensorAllocated(tensor));
}

// MeshBuffer: unallocated — toHost is not yet implemented, skip gracefully.
TEST_F(IsTensorAllocatedTest, ReturnsFalseForMeshBuffer) {
  MeshBuffer nullMeshBuffer;
  auto handle = std::make_shared<MetalTensor>(nullMeshBuffer);
  Tensor tensor(std::static_pointer_cast<void>(handle), nullptr,
                DeviceRuntime::TTMetal);

  EXPECT_FALSE(tt::runtime::isTensorAllocated(tensor));
}
