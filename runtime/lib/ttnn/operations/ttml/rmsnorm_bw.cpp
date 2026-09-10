// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/rmsnorm_bw.h"
#include "metal/ops/rmsnorm_bw/rmsnorm_bw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::RMSNormBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &gamma =
      tensorPool.getTTNNTensorAndValidate(op->gamma());
  const ::ttnn::Tensor &rms = tensorPool.getTTNNTensorAndValidate(op->rms());
  const ::ttnn::Tensor &gradOutput =
      tensorPool.getTTNNTensorAndValidate(op->grad_output());

  std::vector<std::optional<::ttnn::Tensor>> result =
      ::ttml::metal::rmsnorm_bw(input, gamma, rms, gradOutput);

  LOG_ASSERT(result.size() == 2, "rmsnorm_bw expected 2 results, got {}",
             result.size());
  LOG_ASSERT(result.at(0).has_value(), "rmsnorm_bw grad_input not returned");
  LOG_ASSERT(result.at(1).has_value(), "rmsnorm_bw grad_gamma not returned");

  tensorPool.insertTTNNTensorAndValidate(op->grad_input(),
                                         result.at(0).value());
  tensorPool.insertTTNNTensorAndValidate(op->grad_gamma(),
                                         result.at(1).value());
}

} // namespace tt::runtime::ttnn::operations::ttml
