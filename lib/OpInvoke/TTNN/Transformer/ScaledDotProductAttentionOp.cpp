// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/ScaledDotProductAttentionOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

ScaledDotProductAttentionResolvedParams resolveScaledDotProductAttentionParams(
    const ::tt::target::ttnn::ScaledDotProductAttentionOpT &opT) {
  ScaledDotProductAttentionResolvedParams params;
  if (opT.scale.has_value()) {
    params.scale = *opT.scale;
  }
  if (opT.sliding_window_size.has_value()) {
    params.slidingWindowSize = *opT.sliding_window_size;
  }

  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createScaledDotProductAttentionTuple(
    Tag tag, const ::tt::target::ttnn::ScaledDotProductAttentionOpT &opT,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> attentionSink,
    const ScaledDotProductAttentionResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(query, tag), resolveTensorArg(key, tag),
      resolveTensorArg(value, tag),
      attentionMask ? std::make_optional(resolveTensorArg(*attentionMask, tag))
                    : std::nullopt,
      opT.is_causal, params.scale, params.slidingWindowSize,
      params.outputMemoryConfig,
      /*program_config=*/std::nullopt,
      /*compute_kernel_config=*/std::nullopt,
      attentionSink ? std::make_optional(resolveTensorArg(*attentionSink, tag))
                    : std::nullopt);
}

ScaledDotProductAttentionOpResult callScaledDotProductAttention(
    CallType callType,
    const ::tt::target::ttnn::ScaledDotProductAttentionOpT &opT,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice *device) {
  ScaledDotProductAttentionResolvedParams params =
      resolveScaledDotProductAttentionParams(opT);

  auto makeTuple = [&](auto tag) {
    return createScaledDotProductAttentionTuple(tag, opT, query, key, value,
                                                attentionMask, attentionSink,
                                                params);
  };

  callOp(::ttnn::transformer::scaled_dot_product_attention);
}

} // namespace ttnn_op_invoke
