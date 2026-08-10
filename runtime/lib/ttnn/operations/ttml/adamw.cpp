// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/adamw.h"
#include "metal/common/const_utils.hpp"     // ttml::metal::StochasticRounding
#include "metal/optimizers/adamw/adamw.hpp" // ttml::metal::adamw

namespace tt::runtime::ttnn::operations::ttml {

namespace {
// TODO(agobeljic): Remove this once ttml::AdamW accepts beta_pow as tensor.
float scalarValueOf(const ::ttnn::Tensor &tensor, const char *name) {
  LOG_ASSERT(tensor.logical_volume() == 1,
             "AdamW: ", name, " must hold exactly one element, got ",
             tensor.logical_volume());
  const ::ttnn::Tensor host = ::ttnn::from_device(tensor);
  const std::vector<float> values = host.to_vector<float>();
  LOG_ASSERT(!values.empty(), "AdamW: ", name, " read back empty");
  return values.front();
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

  const float beta1Pow = scalarValueOf(
      tensorPool.getTTNNTensorAndValidate(op->beta1_pow()), "beta1_pow");
  const float beta2Pow = scalarValueOf(
      tensorPool.getTTNNTensorAndValidate(op->beta2_pow()), "beta2_pow");

  // Optional AMSGrad max second moment. Its presence enables amsgrad in ttml.
  std::optional<::ttnn::Tensor> maxExpAvgSq = std::nullopt;
  if (op->max_exp_avg_sq()) {
    maxExpAvgSq = tensorPool.getTTNNTensorAndValidate(op->max_exp_avg_sq());
  }

  const ::ttml::metal::StochasticRounding stochasticRounding =
      op->stochastic_rounding() ? ::ttml::metal::StochasticRounding::Enabled
                                : ::ttml::metal::StochasticRounding::Disabled;

  // param, exp_avg, exp_avg_sq (and max_exp_avg_sq) are all updated in place.
  ::ttml::metal::adamw(param, grad, expAvg, expAvgSq, maxExpAvgSq, op->lr(),
                       op->beta1(), op->beta2(), beta1Pow, beta2Pow,
                       op->epsilon(), op->weight_decay(), stochasticRounding);
}

} // namespace tt::runtime::ttnn::operations::ttml
