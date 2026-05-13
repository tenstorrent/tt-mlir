// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_ELTWISE_UNARY_OP_H
#define TTNN_OP_INVOKE_ELTWISE_UNARY_OP_H

#include "operations/eltwise/unary/unary.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace ttnn_op_invoke {

using EltwiseUnaryOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseUnaryResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  bool fastApproxMode = false;
  std::optional<bool> approx;
  std::optional<::ttnn::operations::unary::SigmoidMode> sigmoidMode;
  std::optional<int32_t> vecMode;
  std::optional<float> parameter;
};

EltwiseUnaryResolvedParams resolveEltwiseUnaryParams(
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp);

template <typename Tag>
auto createEltwiseUnaryTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
    TensorArg input, const EltwiseUnaryResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt,
                         /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryOpResult
callEltwiseUnary(CallType callType,
                 const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
                 Fn ttnnOp, TensorArg input,
                 ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params = resolveEltwiseUnaryParams(eltwiseUnaryOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryTuple(tag, eltwiseUnaryOp, input, params);
  };

  return callOp<EltwiseUnaryOpResult>(ttnnOp, callType, makeTuple, device);
}

template <typename Tag>
auto createEltwiseUnaryTanhTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
    TensorArg input, const EltwiseUnaryResolvedParams &params) {
  LOG_ASSERT(params.approx.has_value(), "approx parameter not resolved");
  return std::make_tuple(
      resolveTensorArg(input, tag), params.outputMemoryConfig,
      /*optional_output_tensor=*/std::nullopt, *params.approx,
      /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryOpResult
callEltwiseUnaryTanh(CallType callType,
                     const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
                     Fn ttnnOp, TensorArg input,
                     ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params = resolveEltwiseUnaryParams(eltwiseUnaryOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryTanhTuple(tag, eltwiseUnaryOp, input, params);
  };

  return callOp<EltwiseUnaryOpResult>(ttnnOp, callType, makeTuple, device);
}

template <typename Tag>
auto createEltwiseUnaryWithFastAndApproximateModeTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
    TensorArg input, const EltwiseUnaryResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         /*approx=*/params.fastApproxMode,
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt,
                         /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryOpResult callEltwiseUnaryWithFastAndApproximateMode(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp, Fn ttnnOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params = resolveEltwiseUnaryParams(eltwiseUnaryOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryWithFastAndApproximateModeTuple(
        tag, eltwiseUnaryOp, input, params);
  };

  return callOp<EltwiseUnaryOpResult>(ttnnOp, callType, makeTuple, device);
}

template <typename Tag>
auto createEltwiseUnarySigmoidTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
    TensorArg input, const EltwiseUnaryResolvedParams &params) {
  LOG_ASSERT(params.vecMode.has_value() && params.sigmoidMode.has_value(),
             "sigmoid params (vecMode, sigmoidMode) not resolved");
  return std::make_tuple(resolveTensorArg(input, tag), *params.vecMode,
                         *params.sigmoidMode, params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt,
                         /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryOpResult callEltwiseUnarySigmoid(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp, Fn ttnnOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params = resolveEltwiseUnaryParams(eltwiseUnaryOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnarySigmoidTuple(tag, eltwiseUnaryOp, input, params);
  };

  return callOp<EltwiseUnaryOpResult>(ttnnOp, callType, makeTuple, device);
}

template <typename Tag>
auto createEltwiseUnaryWithFloatParameterTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp,
    TensorArg input, const EltwiseUnaryResolvedParams &params) {
  LOG_ASSERT(params.parameter.has_value(), "float parameter not resolved");
  return std::make_tuple(resolveTensorArg(input, tag), *params.parameter,
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt,
                         /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryOpResult callEltwiseUnaryWithFloatParameter(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryOpT &eltwiseUnaryOp, Fn ttnnOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryResolvedParams params = resolveEltwiseUnaryParams(eltwiseUnaryOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryWithFloatParameterTuple(tag, eltwiseUnaryOp, input,
                                                     params);
  };

  return callOp<EltwiseUnaryOpResult>(ttnnOp, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_ELTWISE_UNARY_OP_H
