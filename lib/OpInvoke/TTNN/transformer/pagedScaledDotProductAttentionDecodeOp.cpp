// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/transformer/pagedScaledDotProductAttentionDecodeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

PagedScaledDotProductAttentionDecodeResolvedParams
resolvePagedScaledDotProductAttentionDecodeParams(
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &opT,
    CallType callType, ::ttnn::MeshDevice &targetDevice) {
  PagedScaledDotProductAttentionDecodeResolvedParams params;

  if (opT.scale.has_value()) {
    params.scale = *opT.scale;
  }
  if (opT.sliding_window_size.has_value()) {
    params.slidingWindowSize = *opT.sliding_window_size;
  }

  const auto computeGrid = targetDevice.compute_with_storage_grid_size();
  if (opT.program_config) {
    params.programConfig =
        operations::utils::createSDPAProgramConfig(*opT.program_config);
  } else if (!opT.is_causal) {
    params.programConfig.emplace();
    params.programConfig->k_chunk_size = 32; // Required for non-causal
    params.programConfig->compute_with_storage_grid_size = computeGrid;
  } else if (targetDevice.arch() == ::tt::ARCH::BLACKHOLE) {
    params.programConfig.emplace();
    params.programConfig->q_chunk_size = 0;
    params.programConfig->k_chunk_size = 0;
    params.programConfig->compute_with_storage_grid_size = computeGrid;
    params.programConfig->max_cores_per_head_batch =
        computeGrid.x * computeGrid.y;
  }

  // Blackhole's SDPA decode default approx-exp path fails SFPI compile
  // (tt-metal #40301).
  if (params.programConfig.has_value() &&
      targetDevice.arch() == ::tt::ARCH::BLACKHOLE) {
    params.programConfig->exp_approx_mode = false;
  }

  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out), callType);
  }

  return params;
}

template <typename Tag>
auto createPagedScaledDotProductAttentionDecodeTuple(
    Tag tag,
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, TensorArg value, TensorArg pageTable,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink,
    const PagedScaledDotProductAttentionDecodeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(query, tag), resolveTensorArg(key, tag),
      resolveTensorArg(value, tag), resolveTensorArg(pageTable, tag),
      opT.is_causal,
      attentionMask ? std::make_optional(resolveTensorArg(*attentionMask, tag))
                    : std::nullopt,
      curPosTensor ? std::make_optional(resolveTensorArg(*curPosTensor, tag))
                   : std::nullopt,
      attentionSink ? std::make_optional(resolveTensorArg(*attentionSink, tag))
                    : std::nullopt,
      params.scale, params.slidingWindowSize, params.outputMemoryConfig,
      params.programConfig,
      /*compute_kernel_config=*/std::nullopt,
      /*block_size_override=*/std::nullopt);
}

PagedScaledDotProductAttentionDecodeOpResult
callPagedScaledDotProductAttentionDecode(
    CallType callType,
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, TensorArg value, TensorArg pageTable,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice &targetDevice) {
  PagedScaledDotProductAttentionDecodeResolvedParams params =
      resolvePagedScaledDotProductAttentionDecodeParams(opT, callType,
                                                        targetDevice);

  auto makeTuple = [&](auto tag) {
    return createPagedScaledDotProductAttentionDecodeTuple(
        tag, opT, query, key, value, pageTable, attentionMask, curPosTensor,
        attentionSink, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(
              ::ttnn::transformer::paged_scaled_dot_product_attention_decode,
              &targetDevice, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(
              ::ttnn::transformer::paged_scaled_dot_product_attention_decode,
              &targetDevice, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::transformer::paged_scaled_dot_product_attention_decode(
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
