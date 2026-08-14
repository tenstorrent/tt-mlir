// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/layernorm_fw.h"
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::LayerNormForwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  // Returns {output, mean-or-nullopt, rstd-or-nullopt}.
  std::vector<std::optional<::ttnn::Tensor>> result =
      ::ttml::metal::layernorm_fw(input, weight, bias, op->epsilon(),
                                  op->return_mean_rstd());

  LOG_ASSERT(result.size() == 3, "layernorm_fw expected 3 results, got {}",
             result.size());
  LOG_ASSERT(result.at(0).has_value(), "layernorm_fw output was not returned");

  tensorPool.insertTTNNTensorAndValidate(op->out(), result.at(0).value());

  if (op->return_mean_rstd()) {
    LOG_ASSERT(result.at(1).has_value(), "mean expected but not returned");
    LOG_ASSERT(result.at(2).has_value(), "rstd expected but not returned");
    tensorPool.insertTTNNTensorAndValidate(op->mean(), result.at(1).value());
    tensorPool.insertTTNNTensorAndValidate(op->rstd(), result.at(2).value());
  }
}

} // namespace tt::runtime::ttnn::operations::ttml
