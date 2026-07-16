// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Integration test: the tt-train (ttml) metal ops are built
// (BUILD_TT_TRAIN=ON), linked into the runtime (libttml.a -> TTRuntimeTTNNOps),
// and runnable on device with their JIT kernels resolving from the tt-metal
// home.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include <xtensor/containers/xarray.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/frobenius_normalize/frobenius_normalize.hpp"

namespace {

class TTMLMetalOpsTest : public ::testing::Test {
protected:
  void SetUp() override { ttml::autograd::ctx().open_device(); }
  void TearDown() override { ttml::autograd::ctx().close_device(); }
};

} // namespace

TEST_F(TTMLMetalOpsTest, FrobeniusNormalizeRunsOnDevice) {
  using namespace ttml;

  constexpr uint32_t H = 32U;
  constexpr uint32_t W = 32U;
  xt::xarray<float> in = xt::ones<float>({1U, 1U, H, W});
  auto input = core::from_xtensor(in, &autograd::ctx().get_device());

  ttnn::Tensor out;
  ASSERT_NO_THROW(out = ttml::metal::frobenius_normalize(input));

  // Ran on device and produced a same-shaped, real (finite) result.
  auto out_xt = core::to_xtensor<float>(out);
  EXPECT_EQ(out_xt.shape(), in.shape());
  EXPECT_TRUE(std::isfinite(out_xt(0, 0, 0, 0)));
}
