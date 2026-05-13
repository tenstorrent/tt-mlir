// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_ELTWISE_BINARY_COMPOSITE_OP_H
#define TTNN_OP_INVOKE_ELTWISE_BINARY_COMPOSITE_OP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

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

template <typename Tag>
auto createEltwiseBinaryCompositeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOpT,
    TensorArg lhs, TensorArg rhs, ::ttnn::MeshDevice *device,
    const EltwiseBinaryCompositeResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
                         params.outputMemoryConfig);
}

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

  auto makeTuple = [&](auto tag) {
    return createEltwiseBinaryCompositeTuple(tag, eltwiseBinaryCompositeOpT,
                                             lhs, rhs, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_constraints(
              eltwiseBinaryCompositeOp, device,
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_runtime(
              eltwiseBinaryCompositeOp, device,
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return eltwiseBinaryCompositeOp(
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
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

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_ELTWISE_BINARY_COMPOSITE_OP_H
