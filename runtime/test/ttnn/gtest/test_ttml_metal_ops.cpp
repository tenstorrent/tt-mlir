// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Integration test: the tt-train (ttml) metal ops are built.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <tt-metalium/shape.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/types.hpp"

#include "metal/ops/frobenius_normalize/frobenius_normalize.hpp"

namespace {

class TTMLMetalOpsTest : public ::testing::Test {
protected:
  void SetUp() override { device_ = ttnn::open_mesh_device(/*device_id=*/0); }
  void TearDown() override {
    if (device_) {
      ttnn::close_device(*device_);
      device_.reset();
    }
  }
  std::shared_ptr<ttnn::MeshDevice> device_;
};

} // namespace

TEST_F(TTMLMetalOpsTest, FrobeniusNormalizeRunsOnDevice) {
  const ttnn::Shape shape({1, 1, 32, 32});
  const ttnn::Tensor input =
      ttnn::ones(shape, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

  ttnn::Tensor out;
  ASSERT_NO_THROW(out = ttml::metal::frobenius_normalize(input));

  // Ran on device and produced a same-shaped, real result.
  EXPECT_EQ(out.logical_shape(), input.logical_shape());
  const std::vector<float> values = ttnn::from_device(out).to_vector<float>();
  ASSERT_FALSE(values.empty());
  EXPECT_TRUE(std::isfinite(values.front()));
}
