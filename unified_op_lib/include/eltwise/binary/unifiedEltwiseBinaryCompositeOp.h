// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_BINARY_COMPOSITE_OP_H
#define UNIFIED_OP_LIB_ELTWISE_BINARY_COMPOSITE_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using EltwiseBinaryCompositeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

using EltwiseBinaryCompositeScalarOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseBinaryCompositeResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

EltwiseBinaryCompositeResolvedParams resolveEltwiseBinaryCompositeParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOpT,
    CallType callType);

template <typename Fn>
EltwiseBinaryCompositeOpResult callEltwiseBinaryComposite(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOpT,
    Fn eltwiseBinaryCompositeOp, TensorArg lhs, TensorArg rhs,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt) {

  EltwiseBinaryCompositeResolvedParams params =
      resolveEltwiseBinaryCompositeParams(eltwiseBinaryCompositeOpT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return ::ttnn::graph::query_op_constraints(
        eltwiseBinaryCompositeOp, device, std::get<::ttnn::TensorSpec>(lhs),
        std::get<::ttnn::TensorSpec>(rhs), params.outputMemoryConfig);
  case CallType::QUERY_OP_RUNTIME:
    return ::ttnn::graph::query_op_runtime(
        eltwiseBinaryCompositeOp, device, std::get<::ttnn::TensorSpec>(lhs),
        std::get<::ttnn::TensorSpec>(rhs), params.outputMemoryConfig);
  case CallType::EXECUTE: {
    const auto &a = *std::get<const ::ttnn::Tensor *>(lhs);
    const auto &b = *std::get<const ::ttnn::Tensor *>(rhs);

    return eltwiseBinaryCompositeOp(a, b, params.outputMemoryConfig);
  }
  }
}

struct EltwiseBinaryCompositeScalarResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::variant<float, int32_t> exponent;
};

EltwiseBinaryCompositeScalarResolvedParams
resolveEltwiseBinaryCompositeScalarParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT,
    CallType callType);

EltwiseBinaryCompositeScalarOpResult callEltwiseBinaryCompositeScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_BINARY_COMPOSITE_OP_H
