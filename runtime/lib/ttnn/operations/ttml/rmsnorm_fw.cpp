// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/rmsnorm_fw.h"
#include "metal/ops/rmsnorm_fw/rmsnorm_fw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::RMSNormForwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &gamma =
      tensorPool.getTTNNTensorAndValidate(op->gamma());

  std::vector<std::optional<::ttnn::Tensor>> result = ::ttml::metal::rmsnorm_fw(
      input, gamma, op->return_intermediates(), op->epsilon());

  LOG_ASSERT(result.size() == 2, "rmsnorm_fw expected 2 results, got {}",
             result.size());
  LOG_ASSERT(result.at(0).has_value(), "rmsnorm_fw output was not returned");

  tensorPool.insertTTNNTensorAndValidate(op->out(), result.at(0).value());

  if (op->return_intermediates()) {
    LOG_ASSERT(result.at(1).has_value(), "rms expected but not returned");
    tensorPool.insertTTNNTensorAndValidate(op->rms(), result.at(1).value());
  }
}

} // namespace tt::runtime::ttnn::operations::ttml
