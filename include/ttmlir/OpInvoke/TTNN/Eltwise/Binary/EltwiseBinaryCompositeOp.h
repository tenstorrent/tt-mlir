// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_ELTWISE_BINARY_ELTWISEBINARYCOMPOSITEOP_H
#define TTMLIR_OPINVOKE_TTNN_ELTWISE_BINARY_ELTWISEBINARYCOMPOSITEOP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"

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
        &eltwiseBinaryCompositeOp);

template <typename Tag>
auto createEltwiseBinaryCompositeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOp,
    TensorArg lhs, TensorArg rhs,
    const EltwiseBinaryCompositeResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
                         params.outputMemoryConfig);
}

template <typename Fn>
EltwiseBinaryCompositeOpResult callEltwiseBinaryComposite(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOp,
    Fn opFn, TensorArg lhs, TensorArg rhs, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt) {

  EltwiseBinaryCompositeResolvedParams params =
      resolveEltwiseBinaryCompositeParams(eltwiseBinaryCompositeOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseBinaryCompositeTuple(tag, eltwiseBinaryCompositeOp, lhs,
                                             rhs, params);
  };

  return callOp<EltwiseBinaryCompositeOpResult>(opFn, callType, makeTuple,
                                                device);
}

struct EltwiseBinaryCompositeScalarResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::variant<float, int32_t> exponent;
};

EltwiseBinaryCompositeScalarResolvedParams
resolveEltwiseBinaryCompositeScalarParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOp);

EltwiseBinaryCompositeScalarOpResult callEltwiseBinaryCompositeScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOp,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_ELTWISE_BINARY_ELTWISEBINARYCOMPOSITEOP_H
