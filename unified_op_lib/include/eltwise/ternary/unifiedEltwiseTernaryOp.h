// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_TERNARY_OP_H
#define UNIFIED_OP_LIB_ELTWISE_TERNARY_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using EltwiseTernaryOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseTernaryResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

EltwiseTernaryResolvedParams resolveEltwiseTernaryParams(
    const tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryOpT);

template <typename Fn>
EltwiseTernaryOpResult callEltwiseTernary(
    CallType callType,
    const tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryWhereOpT,
    Fn eltwiseTernaryOp, TensorArg first, TensorArg second, TensorArg third,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt) {

  EltwiseTernaryResolvedParams params =
      resolveEltwiseTernaryParams(eltwiseTernaryWhereOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    return ::ttnn::graph::query_op_constraints(
        eltwiseTernaryOp, device, std::get<::ttnn::TensorSpec>(first),
        std::get<::ttnn::TensorSpec>(second),
        std::get<::ttnn::TensorSpec>(third), params.outputMemoryConfig);
  }
  case CallType::QUERY_OP_RUNTIME: {
    return ::ttnn::graph::query_op_runtime(
        eltwiseTernaryOp, device, std::get<::ttnn::TensorSpec>(first),
        std::get<::ttnn::TensorSpec>(second),
        std::get<::ttnn::TensorSpec>(third), params.outputMemoryConfig);
  }
  case CallType::EXECUTE: {
    const auto &input_first = *std::get<const ::ttnn::Tensor *>(first);
    const auto &input_second = *std::get<const ::ttnn::Tensor *>(second);
    const auto &input_third = *std::get<const ::ttnn::Tensor *>(third);

    return eltwiseTernaryOp(input_first, input_second, input_third,
                            params.outputMemoryConfig);
  }
  }
}

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_TERNARY__OP_H
