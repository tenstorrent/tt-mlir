// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/ternary/unifiedEltwiseTernaryOp.h"
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

// template <>
EltwiseTernaryResolvedParams resolveEltwiseTernaryParams(
    const ::tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryOpT,
    CallType callType) {

  EltwiseTernaryResolvedParams params;

  if (eltwiseTernaryOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseTernaryOpT.out),
        callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseTernaryOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

// template <typename Fn>
// EltwiseTernaryOpResult callEltwiseTernary(
//     CallType callType,
//     const ::tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryWhereOpT,
//     Fn eltwiseTernaryOp, TensorArg first, TensorArg second, TensorArg third,
//     ::ttnn::MeshDevice *device = nullptr,
//     std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt) {

//   EltwiseTernaryResolvedParams params =
//       resolveEltwiseTernaryParams(eltwiseTernaryWhereOpT);
//   if (outputMemoryConfig.has_value()) {
//     params.outputMemoryConfig = outputMemoryConfig;
//   }

//   switch (callType) {
//   case CallType::QUERY_OP_CONSTRAINTS: {
//     return ::ttnn::graph::query_op_constraints(
//         eltwiseTernaryOp, device, std::get<::ttnn::TensorSpec>(first),
//         std::get<::ttnn::TensorSpec>(second),
//         std::get<::ttnn::TensorSpec>(third), params.outputMemoryConfig);
//   }
//   case CallType::QUERY_OP_RUNTIME: {
//     return ::ttnn::graph::query_op_runtime(
//         eltwiseTernaryOp, device, std::get<::ttnn::TensorSpec>(first),
//         std::get<::ttnn::TensorSpec>(second),
//         std::get<::ttnn::TensorSpec>(third), params.outputMemoryConfig);
//   }
//   case CallType::EXECUTE: {
//     const auto &input_first = *std::get<const ::ttnn::Tensor *>(first);
//     const auto &input_second = *std::get<const ::ttnn::Tensor *>(second);
//     const auto &input_third = *std::get<const ::ttnn::Tensor *>(third);

//     return eltwiseTernaryOp(input_first, input_second, input_third,
//                             params.outputMemoryConfig);
//   }
//   }
// }

} // namespace unifiedOpLib
