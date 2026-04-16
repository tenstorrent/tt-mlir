// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_UNARY_COMPOSITE_OP_H
#define UNIFIED_OP_LIB_ELTWISE_UNARY_COMPOSITE_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using EltwiseUnaryCompositeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseUnaryCompositeResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  bool fastApproxMode = false;
};

EltwiseUnaryCompositeResolvedParams resolveEltwiseUnaryCompositeParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT);

template <typename Fn>
EltwiseUnaryCompositeOpResult callEltwiseUnaryComposite(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    Fn ttnnOp, TensorArg input, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt) {

  EltwiseUnaryCompositeResolvedParams params =
      resolveEltwiseUnaryCompositeParams(eltwiseUnaryCompositeOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return ::ttnn::graph::query_op_constraints(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input),
        params.outputMemoryConfig);
  case CallType::QUERY_OP_RUNTIME:
    return ::ttnn::graph::query_op_runtime(ttnnOp, device,
                                           std::get<::ttnn::TensorSpec>(input),
                                           params.outputMemoryConfig);
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);

    return ttnnOp(in, params.outputMemoryConfig,
                  /*optional_output_tensor=*/std::nullopt,
                  /*sub_core_grids=*/std::nullopt);
  }
  }
}

template <typename Fn>
EltwiseUnaryCompositeOpResult
callEltwiseUnaryCompositeWithFastAndApproximateMode(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    Fn ttnnOp, TensorArg input, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt) {

  EltwiseUnaryCompositeResolvedParams params =
      resolveEltwiseUnaryCompositeParams(eltwiseUnaryCompositeOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return ::ttnn::graph::query_op_constraints(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input),
        params.fastApproxMode, params.outputMemoryConfig);
  case CallType::QUERY_OP_RUNTIME:
    return ::ttnn::graph::query_op_runtime(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input),
        params.fastApproxMode, params.outputMemoryConfig);
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);

    return ttnnOp(in, /*approx=*/params.fastApproxMode,
                  params.outputMemoryConfig,
                  /*optional_output_tensor=*/std::nullopt,
                  /*sub_core_grids=*/std::nullopt);
  }
  }
}

struct EltwiseUnaryCompositeClampScalarResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::variant<float, int32_t> min;
  std::variant<float, int32_t> max;
};

struct EltwiseUnaryCompositeClampTensorResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

EltwiseUnaryCompositeClampScalarResolvedParams
resolveEltwiseUnaryCompositeClampScalarParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT);

EltwiseUnaryCompositeClampTensorResolvedParams
resolveEltwiseUnaryCompositeClampTensorParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT);

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt);

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt);

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_UNARY_COMPOSITE_OP_H
