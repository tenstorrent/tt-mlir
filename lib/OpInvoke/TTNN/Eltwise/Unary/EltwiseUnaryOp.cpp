// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Unary/EltwiseUnaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

EltwiseUnaryResolvedParams resolveEltwiseUnaryParams(
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp) {

  EltwiseUnaryResolvedParams params;

  params.fastApproxMode = false;

  if (eltwiseUnaryOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseUnaryOp.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseUnaryOp.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  if (eltwiseUnaryOp.type == ::tt::target::ttnn::EltwiseUnaryOpType::Tanh) {
    params.approx = false;
  }

  if (eltwiseUnaryOp.type == ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid) {
    params.sigmoidMode = ::ttnn::operations::unary::SigmoidMode::ACCURATE;
    params.vecMode =
        static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
  }

  if (eltwiseUnaryOp.type ==
      ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu) {
    params.parameter =
        eltwiseUnaryOp.params.AsEltwiseOpWithFloatParams()->parameter;
  }

  return params;
}

} // namespace ttnn_op_invoke
