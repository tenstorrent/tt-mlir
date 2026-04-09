// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_BINARY_OP_H
#define UNIFIED_OP_LIB_ELTWISE_BINARY_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using EltwiseBinaryOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseBinaryResolvedParams {
  ::ttnn::DataType outputDataType;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
};

EltwiseBinaryResolvedParams resolveEltwiseBinaryParams(
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT);

template <typename Fn>
EltwiseBinaryOpResult callEltwiseBinary(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT,
    Fn eltwiseBinaryOp, TensorArg lhs, TensorArg rhs,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt) {

  EltwiseBinaryResolvedParams params =
      resolveEltwiseBinaryParams(eltwiseBinaryOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }
  if (outputDType.has_value()) {
    params.outputDType = outputDType;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return ::ttnn::graph::query_op_constraints(
        eltwiseBinaryOp, device, std::get<::ttnn::TensorSpec>(lhs),
        std::get<::ttnn::TensorSpec>(rhs), params.outputDType,
        params.outputMemoryConfig);
  case CallType::QUERY_OP_RUNTIME:
    return ::ttnn::graph::query_op_runtime(
        eltwiseBinaryOp, device, std::get<::ttnn::TensorSpec>(lhs),
        std::get<::ttnn::TensorSpec>(rhs), params.outputDType,
        params.outputMemoryConfig);
  case CallType::EXECUTE: {
    const auto &a = *std::get<const ::ttnn::Tensor *>(lhs);
    const auto &b = *std::get<const ::ttnn::Tensor *>(rhs);

    return eltwiseBinaryOp(a, b, params.outputDType, params.outputMemoryConfig);
  }
  }
}

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_BINARY_OP_H
