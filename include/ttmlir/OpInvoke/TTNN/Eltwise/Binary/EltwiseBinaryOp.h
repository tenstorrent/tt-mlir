// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_ELTWISE_BINARY_ELTWISEBINARYOP_H
#define TTMLIR_OPINVOKE_TTNN_ELTWISE_BINARY_ELTWISEBINARYOP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"

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
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOp);

template <typename Tag>
auto createEltwiseBinaryTuple(
    Tag tag, const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOp,
    TensorArg lhs, TensorArg rhs, const EltwiseBinaryResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
                         params.outputDType, params.outputMemoryConfig);
}

template <typename Fn>
EltwiseBinaryOpResult
callEltwiseBinary(CallType callType,
                  const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOp,
                  Fn opFn, TensorArg lhs, TensorArg rhs,
                  ::ttnn::MeshDevice *device = nullptr) {

  EltwiseBinaryResolvedParams params =
      resolveEltwiseBinaryParams(eltwiseBinaryOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseBinaryTuple(tag, eltwiseBinaryOp, lhs, rhs, params);
  };

  return callOp<EltwiseBinaryOpResult>(opFn, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_ELTWISE_BINARY_ELTWISEBINARYOP_H
