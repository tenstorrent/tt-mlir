// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/adamw.h"
#include "metal/common/const_utils.hpp"     // ttml::metal::StochasticRounding
#include "metal/optimizers/adamw/adamw.hpp" // ttml::metal::adamw_tensor_scalars

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::AdamWOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &param =
      tensorPool.getTTNNTensorAndValidate(op->param());
  const ::ttnn::Tensor &grad = tensorPool.getTTNNTensorAndValidate(op->grad());
  const ::ttnn::Tensor &expAvg =
      tensorPool.getTTNNTensorAndValidate(op->exp_avg());
  const ::ttnn::Tensor &expAvgSq =
      tensorPool.getTTNNTensorAndValidate(op->exp_avg_sq());

  // The step-varying scalars stay on device: ttml reads them in the kernel, so
  // there is no host sync and the op is trace-safe.
  const ::ttnn::Tensor &stepSize =
      tensorPool.getTTNNTensorAndValidate(op->step_size());
  const ::ttnn::Tensor &invSqrtBc2 =
      tensorPool.getTTNNTensorAndValidate(op->inv_sqrt_bc2());
  const ::ttnn::Tensor &decayFactor =
      tensorPool.getTTNNTensorAndValidate(op->decay_factor());

  // Optional AMSGrad max second moment. Its presence enables amsgrad in ttml.
  std::optional<::ttnn::Tensor> maxExpAvgSq = std::nullopt;
  if (op->max_exp_avg_sq()) {
    maxExpAvgSq = tensorPool.getTTNNTensorAndValidate(op->max_exp_avg_sq());
  }

  const auto stochasticRounding =
      op->stochastic_rounding() ? ::ttml::metal::StochasticRounding::Enabled
                                : ::ttml::metal::StochasticRounding::Disabled;

  // param, exp_avg, exp_avg_sq (and max_exp_avg_sq) are all updated in place.
  ::ttml::metal::adamw_tensor_scalars(
      param, grad, expAvg, expAvgSq, maxExpAvgSq, stepSize, invSqrtBc2,
      decayFactor, op->beta1(), op->beta2(), op->epsilon(), stochasticRounding);
}

} // namespace tt::runtime::ttnn::operations::ttml
