// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/layernorm_bw.h"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::LayerNormBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &gamma =
      tensorPool.getTTNNTensorAndValidate(op->gamma());
  const ::ttnn::Tensor &mean = tensorPool.getTTNNTensorAndValidate(op->mean());
  const ::ttnn::Tensor &rstd = tensorPool.getTTNNTensorAndValidate(op->rstd());
  const ::ttnn::Tensor &dL_dout =
      tensorPool.getTTNNTensorAndValidate(op->dl_dout());

  std::vector<std::optional<::ttnn::Tensor>> result =
      ::ttml::metal::layernorm_bw(input, gamma, mean, rstd, dL_dout);
  LOG_ASSERT(result.size() == 3, "layernorm_bw expected 3 results, got {}",
             result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    LOG_ASSERT(result[i].has_value(), "layernorm_bw result {} was not returned",
               i);
  }

  tensorPool.insertTTNNTensorAndValidate(op->dx(), result[0].value());
  tensorPool.insertTTNNTensorAndValidate(op->dgamma(), result[1].value());
  tensorPool.insertTTNNTensorAndValidate(op->dbeta(), result[2].value());
}

} // namespace tt::runtime::ttnn::operations::ttml
