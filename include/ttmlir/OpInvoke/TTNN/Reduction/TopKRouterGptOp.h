// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_REDUCTION_TOPKROUTERGPTOP_H
#define TTMLIR_OPINVOKE_TTNN_REDUCTION_TOPKROUTERGPTOP_H

#include "operations/experimental/topk_router_gpt/topk_router_gpt.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tuple>

namespace ttnn_op_invoke {

using TopKRouterGptOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse,
                 std::tuple<::ttnn::Tensor, ::ttnn::Tensor>>;

struct TopKRouterGptResolvedParams {
  uint32_t k;
  uint32_t numExperts;
};

TopKRouterGptResolvedParams
resolveTopKRouterGptParams(const ::tt::target::ttnn::TopKRouterGptOpT &op);

TopKRouterGptOpResult
callTopKRouterGpt(CallType callType,
                  const ::tt::target::ttnn::TopKRouterGptOpT &op,
                  TensorArg input, TensorArg weight, TensorArg bias,
                  ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_REDUCTION_TOPKROUTERGPTOP_H
