// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test: isTensorAllocated for TTMetal tensors must not crash.
//
// Background: builder_runtime.py calls is_allocated() before to_host() to
// skip unallocated const-eval output tensors. The TTMetal implementation of
// isTensorAllocated was previously a LOG_FATAL stub, which caused every
// ttmetal-target test to fail when the workaround was introduced.

#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using tt::runtime::DeviceRuntime;
using tt::runtime::Tensor;
using tt::runtime::TensorDesc;
using tt::runtime::ttmetal::HostBuffer;
using tt::runtime::ttmetal::MetalTensor;

class IsTensorAllocatedTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(DeviceRuntime::TTMetal);
  }
};

// A tensor whose MetalTensor variant holds only a TensorDesc (no buffer) is
// unallocated. This is the case a const-eval placeholder produces.
TEST_F(IsTensorAllocatedTest, ReturnsFalseForTensorDesc) {
  TensorDesc desc({1, 4}, tt::target::DataType::Float32);
  auto handle = std::make_shared<MetalTensor>(desc);
  Tensor tensor(std::static_pointer_cast<void>(handle), nullptr,
                DeviceRuntime::TTMetal);

  EXPECT_FALSE(tt::runtime::isTensorAllocated(tensor));
}

// A tensor whose MetalTensor variant holds a HostBuffer is allocated.
// The buffer pointer itself may be null; only the variant alternative matters.
TEST_F(IsTensorAllocatedTest, ReturnsTrueForHostBuffer) {
  HostBuffer nullBuffer; // null shared_ptr<tt::tt_metal::HostBuffer>
  auto handle = std::make_shared<MetalTensor>(nullBuffer);
  Tensor tensor(std::static_pointer_cast<void>(handle), nullptr,
                DeviceRuntime::TTMetal);

  EXPECT_TRUE(tt::runtime::isTensorAllocated(tensor));
}
