// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_UNARY_OP_H
#define UNIFIED_OP_LIB_ELTWISE_UNARY_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <cstdint>
#include <operations/eltwise/unary_ng/unary_ng.hpp>
#include <optional>
#include <ttnn/tensor/tensor.hpp>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using EltwiseUnaryOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseUnaryResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  bool fastApproxMode = false;
  bool approx = false;
  ::ttnn::operations::unary::SigmoidMode sigmoidMode;
  int32_t vecMode;
  float parameter;
};

EltwiseUnaryResolvedParams resolveEltwiseUnaryParams(
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT,
    CallType callType);

template <typename Fn>
EltwiseUnaryOpResult
callEltwiseUnary(CallType callType,
                 const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT,
                 Fn ttnnOp, TensorArg input,
                 ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params =
      resolveEltwiseUnaryParams(eltwiseUnaryOpT, callType);

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
EltwiseUnaryOpResult
callEltwiseUnaryTanh(CallType callType,
                     const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT,
                     Fn ttnnOp, TensorArg input,
                     ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params =
      resolveEltwiseUnaryParams(eltwiseUnaryOpT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return ::ttnn::graph::query_op_constraints(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input),
        params.outputMemoryConfig, /*optional_output_tensor=*/std::nullopt,
        params.approx,
        /*sub_core_grids=*/std::nullopt);
  case CallType::QUERY_OP_RUNTIME:
    return ::ttnn::graph::query_op_runtime(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input),
        params.outputMemoryConfig, /*optional_output_tensor=*/std::nullopt,
        params.approx,
        /*sub_core_grids=*/std::nullopt);
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);

    return ttnnOp(in, params.outputMemoryConfig,
                  /*optional_output_tensor=*/std::nullopt, params.approx,
                  /*sub_core_grids=*/std::nullopt);
  }
  }
}

template <typename Fn>
EltwiseUnaryOpResult callEltwiseUnaryWithFastAndApproximateMode(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT, Fn ttnnOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params =
      resolveEltwiseUnaryParams(eltwiseUnaryOpT, callType);

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

template <typename Fn>
EltwiseUnaryOpResult callEltwiseUnarySigmoid(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT, Fn ttnnOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params =
      resolveEltwiseUnaryParams(eltwiseUnaryOpT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    return ::ttnn::graph::query_op_constraints(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input), params.vecMode,
        params.sigmoidMode, params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt,
        /*sub_core_grids=*/std::nullopt);
  }
  case CallType::QUERY_OP_RUNTIME: {
    return ::ttnn::graph::query_op_runtime(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input), params.vecMode,
        params.sigmoidMode, params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt,
        /*sub_core_grids=*/std::nullopt);
  }
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);

    return ttnnOp(in, params.vecMode, params.sigmoidMode,
                  params.outputMemoryConfig,
                  /*optional_output_tensor=*/std::nullopt,
                  /*sub_core_grids=*/std::nullopt);
  }
  }
}

template <typename Fn>
EltwiseUnaryOpResult callEltwiseUnaryWithFloatParameter(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOpT, Fn ttnnOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params =
      resolveEltwiseUnaryParams(eltwiseUnaryOpT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    return ::ttnn::graph::query_op_constraints(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input), params.parameter,
        params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt,
        /*sub_core_grids=*/std::nullopt);
  }
  case CallType::QUERY_OP_RUNTIME: {
    return ::ttnn::graph::query_op_runtime(
        ttnnOp, device, std::get<::ttnn::TensorSpec>(input), params.parameter,
        params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt,
        /*sub_core_grids=*/std::nullopt);
  }
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);

    return ttnnOp(in, params.parameter, params.outputMemoryConfig,
                  /*optional_output_tensor=*/std::nullopt,
                  /*sub_core_grids=*/std::nullopt);
  }
  }
}

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_UNARY_OP_H
