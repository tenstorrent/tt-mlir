// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/transformer/scaledDotProductAttentionDecodeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>
#include <vector>

namespace ttnn_op_invoke {

ScaledDotProductAttentionDecodeResolvedParams
resolveScaledDotProductAttentionDecodeParams(
    const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &opT,
    CallType callType) {
  ScaledDotProductAttentionDecodeResolvedParams params;

  // The current position information is required for this op. It can either be
  // passed as a tensor or as a uint vector. The uint vector is not wrapped in a
  // std::optional so we must pass an empty vector.
  params.curPos = {};

  if (opT.scale.has_value()) {
    params.scale = *opT.scale;
  }
  if (opT.program_config) {
    params.programConfig =
        operations::utils::createSDPAProgramConfig(*opT.program_config);
  }
  
  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createScaledDotProductAttentionDecodeTuple(
    Tag tag,
    const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink,
    const ScaledDotProductAttentionDecodeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(query, tag), resolveTensorArg(key, tag),
      resolveTensorArg(value, tag), opT.is_causal,
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
    const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice &targetDevice) {
  ScaledDotProductAttentionDecodeResolvedParams params =
      resolveScaledDotProductAttentionDecodeParams(opT, callType);

  auto makeTuple = [&](auto tag) {
    return createScaledDotProductAttentionDecodeTuple(
        tag, opT, query, key, value, attentionMask, curPosTensor, attentionSink,
        params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(
              ::ttnn::transformer::scaled_dot_product_attention_decode,
              &targetDevice, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(
              ::ttnn::transformer::scaled_dot_product_attention_decode,
              &targetDevice, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::transformer::scaled_dot_product_attention_decode(
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
