// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/ScaledDotProductAttentionDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"

#include <optional>
#include <variant>
#include <vector>

namespace ttnn_op_invoke {

ScaledDotProductAttentionDecodeResolvedParams
resolveScaledDotProductAttentionDecodeParams(
    const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &op) {
  ScaledDotProductAttentionDecodeResolvedParams params;

  // The current position information is required for this op. It can either be
  // passed as a tensor or as a uint vector. The uint vector is not wrapped in a
  // std::optional so we must pass an empty vector.
  params.curPos = {};

  if (op.scale.has_value()) {
    params.scale = *op.scale;
  }
  if (op.program_config) {
    params.programConfig =
        operations::utils::createSDPAProgramConfig(*op.program_config);
  }

  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createScaledDotProductAttentionDecodeTuple(
    Tag tag, const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &op,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink,
    const ScaledDotProductAttentionDecodeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(query, tag), resolveTensorArg(key, tag),
      resolveTensorArg(value, tag), op.is_causal,
      attentionMask ? std::make_optional(resolveTensorArg(*attentionMask, tag))
                    : std::nullopt,
      params.curPos,
      curPosTensor ? std::make_optional(resolveTensorArg(*curPosTensor, tag))
                   : std::nullopt,
      attentionSink ? std::make_optional(resolveTensorArg(*attentionSink, tag))
                    : std::nullopt,
      params.scale, /*sliding_window_size=*/std::nullopt,
      params.outputMemoryConfig, params.programConfig,
      /*compute_kernel_config=*/std::nullopt,
      /*share_cache=*/std::nullopt);
}

ScaledDotProductAttentionDecodeOpResult callScaledDotProductAttentionDecode(
    CallType callType,
    const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &op,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice *device) {
  ScaledDotProductAttentionDecodeResolvedParams params =
      resolveScaledDotProductAttentionDecodeParams(op);

  auto makeTuple = [&](auto tag) {
    return createScaledDotProductAttentionDecodeTuple(
        tag, op, query, key, value, attentionMask, curPosTensor, attentionSink,
        params);
  };

  return callOp<ScaledDotProductAttentionDecodeOpResult>(
      WRAP_OP(::ttnn::transformer::scaled_dot_product_attention_decode),
      callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
