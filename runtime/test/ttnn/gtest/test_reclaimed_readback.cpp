// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using ::tt::runtime::ttnn::TTNNTensorWrapper;

class ReclaimedReadbackTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
  }

  static tt::runtime::Tensor makeHostTensor() {
    std::vector<float> data = {0, 1, 2, 3};
    std::vector<uint32_t> shape = {2, 2};
    std::vector<int64_t> stride = {2, 1};
    return tt::runtime::createOwnedHostTensor(data.data(), shape, stride,
                                              sizeof(float),
                                              tt::target::DataType::Float32);
  }
};

} // namespace

// A normal (non-reclaimed) tensor copies to host without error.
TEST_F(ReclaimedReadbackTest, NormalTensorCopiesToHost) {
  tt::runtime::Tensor tensor = makeHostTensor();
  EXPECT_NO_THROW(
      tt::runtime::toHost(tensor, /*untilize=*/false, /*blocking=*/true));
}

// A const-eval input whose device buffer was reclaimed (load_cached swapped in
// a placeholder) must not be read back silently: toHost errors instead of
// returning the placeholder's contents.
TEST_F(ReclaimedReadbackTest, ReclaimedTensorReadbackThrows) {
  tt::runtime::Tensor tensor = makeHostTensor();
  tensor.as<TTNNTensorWrapper>(tt::runtime::DeviceRuntime::TTNN)
      .setReclaimed(true);
  EXPECT_THROW(
      tt::runtime::toHost(tensor, /*untilize=*/false, /*blocking=*/true),
      std::runtime_error);
}
