// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/adamw.h"
#include "metal/common/const_utils.hpp"     // ttml::metal::StochasticRounding
#include "metal/optimizers/adamw/adamw.hpp" // ttml::metal::adamw
#include "ttnn/distributed/api.hpp"

namespace tt::runtime::ttnn::operations::ttml {

static float readScalar(const ::ttnn::Tensor &tensor) {
  return ::ttnn::distributed::get_device_tensors(tensor)
      .front()
      .to_vector<float>()
      .front();
}

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

  // param, exp_avg, exp_avg_sq (and max_exp_avg_sq) are all updated in place.
  ::ttml::metal::adamw(
      param, grad, expAvg, expAvgSq, maxExpAvgSq,
      readScalar(tensorPool.getTTNNTensorAndValidate(op->lr())), op->beta1(),
      op->beta2(),
      readScalar(tensorPool.getTTNNTensorAndValidate(op->beta1_pow())),
      readScalar(tensorPool.getTTNNTensorAndValidate(op->beta2_pow())),
      op->epsilon(), op->weight_decay(), stochasticRounding);
}

} // namespace tt::runtime::ttnn::operations::ttml
