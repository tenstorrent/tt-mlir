// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Reduction/TopKRouterGptOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

TopKRouterGptResolvedParams
resolveTopKRouterGptParams(const ::tt::target::ttnn::TopKRouterGptOpT &op) {
  TopKRouterGptResolvedParams params;
  params.k = static_cast<uint32_t>(op.k);
  params.numExperts = static_cast<uint32_t>(op.num_experts);
  return params;
}

template <typename Tag>
auto createTopKRouterGptTuple(
    Tag tag, const ::tt::target::ttnn::TopKRouterGptOpT &topKRouterGptOp,
    TensorArg input, TensorArg weight, TensorArg bias,
    const TopKRouterGptResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag),
      resolveTensorArg(bias, tag), params.k, params.numExperts);
}

TopKRouterGptOpResult
callTopKRouterGpt(CallType callType,
                  const ::tt::target::ttnn::TopKRouterGptOpT &op,
                  TensorArg input, TensorArg weight, TensorArg bias,
                  ::ttnn::MeshDevice *device) {
  TopKRouterGptResolvedParams params = resolveTopKRouterGptParams(op);

  auto makeTuple = [&](auto tag) {
    return createTopKRouterGptTuple(tag, op, input, weight, bias, params);
  };

  return callOp<TopKRouterGptOpResult>(
      WRAP_OP(::ttnn::experimental::topk_router_gpt), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
