// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_ELTWISE_TERNARY_OP_H
#define TTNN_OP_INVOKE_ELTWISE_TERNARY_OP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

using EltwiseTernaryOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseTernaryResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

EltwiseTernaryResolvedParams resolveEltwiseTernaryParams(
    const tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryOp);

template <typename Tag>
auto createEltwiseTernaryTuple(
    Tag tag,
    const tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryWhereOp,
    TensorArg first, TensorArg second, TensorArg third,
    const EltwiseTernaryResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(first, tag), resolveTensorArg(second, tag),
      resolveTensorArg(third, tag), params.outputMemoryConfig);
}

template <typename Fn>
EltwiseTernaryOpResult callEltwiseTernary(
    CallType callType,
    const tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryWhereOp,
    Fn opFn, TensorArg first, TensorArg second, TensorArg third,
    ::ttnn::MeshDevice *device = nullptr) {

  EltwiseTernaryResolvedParams params =
      resolveEltwiseTernaryParams(eltwiseTernaryWhereOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseTernaryTuple(tag, eltwiseTernaryWhereOp, first, second,
                                     third, params);
  };

  return callOp<EltwiseTernaryOpResult>(opFn, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_ELTWISE_TERNARY_OP_H
