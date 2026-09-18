// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/adamw.h"
#include "metal/common/const_utils.hpp"     // ttml::metal::StochasticRounding
#include "metal/optimizers/adamw/adamw.hpp" // ttml::metal::adamw
#include "ttnn/distributed/api.hpp"

#include <random>

namespace tt::runtime::ttnn::operations::ttml {

// Device -> host readback of a single-element tensor. Cached per program run
// in ProgramContext so N AdamW ops sharing one lr tensor cost one sync.
static float readScalar(ProgramContext &context,
                        const ::tt::target::ttnn::TensorRef *ref) {
  return context.getHostScalar(ref->global_id(), [&]() {
    const ::ttnn::Tensor &tensor =
        context.getTensorPool().getTTNNTensorAndValidate(ref);
    LOG_ASSERT(tensor.logical_volume() == 1,
               "AdamW scalar operand must have exactly one element, got ",
               tensor.logical_volume());
    LOG_ASSERT(tensor.dtype() == ::ttnn::DataType::FLOAT32,
               "AdamW scalar operand must be float32");
    const std::vector<::ttnn::Tensor> shards =
        ::ttnn::distributed::get_device_tensors(tensor);
    LOG_ASSERT(!shards.empty(), "AdamW scalar operand has no device shards");
    const std::vector<float> values = shards.front().to_vector<float>();
    LOG_ASSERT(values.size() == 1, "AdamW scalar readback returned ",
               values.size(), " elements");
    return values.front();
  });
}

// Generate a seed for stochastic rounding. `ttml::metal::adamw` requires it iff
// stochastic rounding is enabled. A constant seed would defeat the purpose of
// stochastic rounding by reusing the same rounding sequence in every step.
static uint32_t drawStochasticRoundingSeed() {
  static thread_local std::mt19937 generator(std::random_device{}());
  return static_cast<uint32_t>(generator());
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
  const std::optional<uint32_t> stochasticRoundingSeed =
      op->stochastic_rounding()
          ? std::optional<uint32_t>(drawStochasticRoundingSeed())
          : std::nullopt;

  // param, exp_avg, exp_avg_sq (and max_exp_avg_sq) are all updated in place.
  ::ttml::metal::adamw(
      param, grad, expAvg, expAvgSq, maxExpAvgSq, readScalar(context, op->lr()),
      op->beta1(), op->beta2(), readScalar(context, op->beta1_pow()),
      readScalar(context, op->beta2_pow()), op->epsilon(), op->weight_decay(),
      stochasticRounding, stochasticRoundingSeed);
}

} // namespace tt::runtime::ttnn::operations::ttml
