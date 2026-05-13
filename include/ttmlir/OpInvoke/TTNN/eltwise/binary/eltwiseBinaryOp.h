// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_ELTWISE_BINARY_OP_H
#define TTNN_OP_INVOKE_ELTWISE_BINARY_OP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

using EltwiseBinaryOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseBinaryResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
};

EltwiseBinaryResolvedParams resolveEltwiseBinaryParams(
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT,
    CallType callType);

template <typename Tag>
auto createEltwiseBinaryTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT,
    TensorArg lhs, TensorArg rhs, ::ttnn::MeshDevice *device,
    const EltwiseBinaryResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
                         params.outputDType, params.outputMemoryConfig);
}

template <typename Fn>
EltwiseBinaryOpResult
callEltwiseBinary(CallType callType,
                  const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT,
                  Fn eltwiseBinaryOp, TensorArg lhs, TensorArg rhs,
                  ::ttnn::MeshDevice *device = nullptr) {

  EltwiseBinaryResolvedParams params =
      resolveEltwiseBinaryParams(eltwiseBinaryOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createEltwiseBinaryTuple(tag, eltwiseBinaryOpT, lhs, rhs, device,
                                    params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_constraints(
              eltwiseBinaryOp, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_runtime(
              eltwiseBinaryOp, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return eltwiseBinaryOp(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_ELTWISE_BINARY_OP_H
