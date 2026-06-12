// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Unary/EltwiseUnaryOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>

namespace ttnn_op_invoke {

EltwiseUnaryResolvedParams resolveEltwiseUnaryParams(
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp) {

  EltwiseUnaryResolvedParams params;

  // Preserving the hard-coded constant from runtime.
  params.fastApproxMode = false;

  if (eltwiseUnaryOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseUnaryOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*eltwiseUnaryOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  switch (eltwiseUnaryOp.type) {
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tanh: {
    params.approx = false;
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid: {
    params.sigmoidMode = ::ttnn::operations::unary::SigmoidMode::ACCURATE;
    params.vecMode =
        static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu: {
    params.parameter =
        eltwiseUnaryOp.params.AsEltwiseOpWithFloatParams()->parameter;
    break;
  }
  default:
    break;
  }

  return params;
}

} // namespace ttnn_op_invoke
