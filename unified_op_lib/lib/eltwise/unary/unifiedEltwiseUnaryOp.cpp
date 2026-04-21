// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/unary/unifiedEltwiseUnaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "utils/utils.h"
#include <operations/functions.hpp>
#include <optional>
#include <variant>

namespace unifiedOpLib {

EltwiseUnaryResolvedParams resolveEltwiseUnaryParams(
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT,
    CallType callType) {

  EltwiseUnaryResolvedParams params;

  params.fastApproxMode = true;

  if (eltwiseUnaryOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseUnaryOpT.out),
        callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseUnaryOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  if (eltwiseUnaryOpT.type == ::tt::target::ttnn::EltwiseUnaryOpType::Tanh) {
    params.approx = false;
  }

  if (eltwiseUnaryOpT.type == ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid) {
    params.sigmoidMode = ::ttnn::operations::unary::SigmoidMode::ACCURATE;
    params.vecMode =
        static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
  }

  if (eltwiseUnaryOpT.type ==
      ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu) {
    params.parameter =
        eltwiseUnaryOpT.params.AsEltwiseOpWithFloatParams()->parameter;
  }

  return params;
}

} // namespace unifiedOpLib
