// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_ELTWISE_UNARY_COMPOSITE_OP_H
#define TTNN_OP_INVOKE_ELTWISE_UNARY_COMPOSITE_OP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace ttnn_op_invoke {

using EltwiseUnaryCompositeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseUnaryCompositeResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  bool fastApproxMode = false;
};

EltwiseUnaryCompositeResolvedParams resolveEltwiseUnaryCompositeParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    CallType callType);

template <typename Tag>
auto createEltwiseUnaryCompositeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device,
    const EltwiseUnaryCompositeResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt,
                         /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryCompositeOpResult
callEltwiseUnaryComposite(CallType callType,
                          const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
                              &eltwiseUnaryCompositeOpT,
                          Fn ttnnOp, TensorArg input,
                          ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryCompositeResolvedParams params =
      resolveEltwiseUnaryCompositeParams(eltwiseUnaryCompositeOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeTuple(tag, eltwiseUnaryCompositeOpT,
                                            input, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_constraints(
              ttnnOp, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_runtime(
              ttnnOp, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ttnnOp(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

template <typename Tag>
auto createEltwiseUnaryCompositeWithFastAndApproximateModeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device,
    const EltwiseUnaryCompositeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), /*approx=*/params.fastApproxMode,
      params.outputMemoryConfig, /*optional_output_tensor=*/std::nullopt,
      /*sub_core_grids=*/std::nullopt);
}

template <typename Fn>
EltwiseUnaryCompositeOpResult
callEltwiseUnaryCompositeWithFastAndApproximateMode(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    Fn ttnnOp, TensorArg input, ::ttnn::MeshDevice *device = nullptr) {

  EltwiseUnaryCompositeResolvedParams params =
      resolveEltwiseUnaryCompositeParams(eltwiseUnaryCompositeOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeWithFastAndApproximateModeTuple(
        tag, eltwiseUnaryCompositeOpT, input, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_constraints(
              ttnnOp, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_runtime(
              ttnnOp, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ttnnOp(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

struct EltwiseUnaryCompositeClampScalarResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<std::variant<float, int32_t>> min;
  std::optional<std::variant<float, int32_t>> max;
};

struct EltwiseUnaryCompositeClampTensorResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

EltwiseUnaryCompositeClampScalarResolvedParams
resolveEltwiseUnaryCompositeClampScalarParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    CallType callType);

EltwiseUnaryCompositeClampTensorResolvedParams
resolveEltwiseUnaryCompositeClampTensorParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    CallType callType);

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr);

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max,
    ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_ELTWISE_UNARY_COMPOSITE_OP_H
