// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/adamw.h"
#include "metal/common/const_utils.hpp"     // ttml::metal::StochasticRounding
#include "metal/optimizers/adamw/adamw.hpp" // ttml::metal::adamw

#include <random>

namespace tt::runtime::ttnn::operations::ttml {

namespace {
// ttml requires a seed iff stochastic rounding is enabled, and expects a fresh
// one per step: reusing a seed would round every optimizer step identically,
// which defeats the purpose of stochastic rounding. The flatbuffer carries no
// seed, so draw one here, mirroring what the ttml kernel used to do internally
// before the seed became a caller-supplied argument.
uint32_t drawStochasticRoundingSeed() {
  static thread_local std::mt19937 generator(std::random_device{}());
  return static_cast<uint32_t>(generator());
}
} // namespace

void run(const ::tt::target::ttnn::AdamWOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &param =
      tensorPool.getTTNNTensorAndValidate(op->param());
  const ::ttnn::Tensor &grad = tensorPool.getTTNNTensorAndValidate(op->grad());
  const ::ttnn::Tensor &expAvg =
      tensorPool.getTTNNTensorAndValidate(op->exp_avg());
  const ::ttnn::Tensor &expAvgSq =
      tensorPool.getTTNNTensorAndValidate(op->exp_avg_sq());

  // Optional AMSGrad max second moment. Its presence enables amsgrad in ttml.
  std::optional<::ttnn::Tensor> maxExpAvgSq = std::nullopt;
  if (op->max_exp_avg_sq()) {
    maxExpAvgSq = tensorPool.getTTNNTensorAndValidate(op->max_exp_avg_sq());
  }

  const ::ttml::metal::StochasticRounding stochasticRounding =
      op->stochastic_rounding() ? ::ttml::metal::StochasticRounding::Enabled
                                : ::ttml::metal::StochasticRounding::Disabled;
  const std::optional<uint32_t> stochasticRoundingSeed =
      op->stochastic_rounding()
          ? std::optional<uint32_t>(drawStochasticRoundingSeed())
          : std::nullopt;

  // param, exp_avg, exp_avg_sq (and max_exp_avg_sq) are all updated in place.
  ::ttml::metal::adamw(param, grad, expAvg, expAvgSq, maxExpAvgSq, op->lr(),
                       op->beta1(), op->beta2(), op->beta1_pow(),
                       op->beta2_pow(), op->epsilon(), op->weight_decay(),
                       stochasticRounding, stochasticRoundingSeed);
}

} // namespace tt::runtime::ttnn::operations::ttml
