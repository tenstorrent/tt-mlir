// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/adamw.h"
#include "metal/common/const_utils.hpp"     // ttml::metal::StochasticRounding
#include "metal/optimizers/adamw/adamw.hpp" // ttml::metal::adamw
#include "tt/runtime/detail/ttnn/utils.h"

#include <tt-metalium/tensor/tensor_types.hpp> // tt::tt_metal::is_floating_point

namespace tt::runtime::ttnn::operations::ttml {

namespace {
// Reads a single-element tensor operand back to host as a float for ttml,
// whose API takes the value rather than a tensor. The readback is a blocking
// sync, so it is cached on the program context: every adamw op of a step reads
// the same lr and bias-correction tensors, so a step costs 3 syncs rather than
// 3 per parameter. The cache key includes the tensor's version, so a write
// into the tensor mid-program is not served stale values.
float scalarValueOf(ProgramContext &context,
                    const ::tt::target::ttnn::TensorRef *tensorRef,
                    const char *name) {
  LOG_ASSERT(tensorRef, "AdamW: ", name, " is missing from the flatbuffer");

  const TTNNTensorWrapper &wrapper =
      context.getTensorPool().getTTNNTensorWrapperAndValidate(tensorRef);
  const uint64_t version = wrapper.getVersion();

  if (std::optional<float> cached =
          context.getCachedHostScalar(tensorRef->global_id(), version)) {
    return *cached;
  }

  const ::ttnn::Tensor &tensor = wrapper.getTensor();
  LOG_ASSERT(tensor.logical_volume() == 1, "AdamW: ", name,
             " must hold exactly one element, got ", tensor.logical_volume());
  // Checked here so a non-float dtype names the operand instead of failing
  // inside tt-metal's to_vector<float>.
  LOG_ASSERT(::tt::tt_metal::is_floating_point(tensor.dtype()), "AdamW: ", name,
             " must be a float tensor, got ", static_cast<int>(tensor.dtype()),
             " (::ttnn::DataType)");

  // `to_vector` copies to host itself, so no explicit `from_device` is needed.
  const float value = utils::getScalarFromTensor<float>(tensor);
  context.cacheHostScalar(tensorRef->global_id(), version, value);
  return value;
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

  const float lr = scalarValueOf(context, op->lr(), "lr");
  const float beta1Pow = scalarValueOf(context, op->beta1_pow(), "beta1_pow");
  const float beta2Pow = scalarValueOf(context, op->beta2_pow(), "beta2_pow");

  // Optional AMSGrad max second moment. Its presence enables amsgrad in ttml.
  std::optional<::ttnn::Tensor> maxExpAvgSq = std::nullopt;
  if (op->max_exp_avg_sq()) {
    maxExpAvgSq = tensorPool.getTTNNTensorAndValidate(op->max_exp_avg_sq());
  }

  const ::ttml::metal::StochasticRounding stochasticRounding =
      op->stochastic_rounding() ? ::ttml::metal::StochasticRounding::Enabled
                                : ::ttml::metal::StochasticRounding::Disabled;

  // param, exp_avg, exp_avg_sq (and max_exp_avg_sq) are all updated in place.
  ::ttml::metal::adamw(param, grad, expAvg, expAvgSq, maxExpAvgSq, lr,
                       op->beta1(), op->beta2(), beta1Pow, beta2Pow,
                       op->epsilon(), op->weight_decay(), stochasticRounding);

  // This op mutates tensors in place, so it must keep the host-scalar cache
  // contract itself (see ProgramContext::getCachedHostScalar): bump the
  // version of everything written, so e.g. a single-element param later read
  // as a scalar operand is not served its pre-update value.
  tensorPool.getTTNNTensorWrapperAndValidate(op->param()).updateVersion();
  tensorPool.getTTNNTensorWrapperAndValidate(op->exp_avg()).updateVersion();
  tensorPool.getTTNNTensorWrapperAndValidate(op->exp_avg_sq()).updateVersion();
  if (op->max_exp_avg_sq()) {
    tensorPool.getTTNNTensorWrapperAndValidate(op->max_exp_avg_sq())
        .updateVersion();
  }
}

} // namespace tt::runtime::ttnn::operations::ttml
