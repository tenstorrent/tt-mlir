// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/PagedScaledDotProductAttentionDecodeOp.h"
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
    ::ttnn::MeshDevice *device) {
  PagedScaledDotProductAttentionDecodeResolvedParams params;

  if (opT.scale.has_value()) {
    params.scale = *opT.scale;
  }
  if (opT.sliding_window_size.has_value()) {
    params.slidingWindowSize = *opT.sliding_window_size;
  }

  const auto computeGrid = device->compute_with_storage_grid_size();
  if (opT.program_config) {
    params.programConfig =
        operations::utils::createSDPAProgramConfig(*opT.program_config);
  } else if (!opT.is_causal) {
    params.programConfig.emplace();
    params.programConfig->k_chunk_size = 32; // Required for non-causal
    params.programConfig->compute_with_storage_grid_size = computeGrid;
  } else if (device->arch() == ::tt::ARCH::BLACKHOLE) {
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
      device->arch() == ::tt::ARCH::BLACKHOLE) {
    params.programConfig->exp_approx_mode = false;
  }

  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
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
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice *device) {
  PagedScaledDotProductAttentionDecodeResolvedParams params =
      resolvePagedScaledDotProductAttentionDecodeParams(opT, device);

  auto makeTuple = [&](auto tag) {
    return createPagedScaledDotProductAttentionDecodeTuple(
        tag, opT, query, key, value, pageTable, attentionMask, curPosTensor,
        attentionSink, params);
  };

  return callOp<PagedScaledDotProductAttentionDecodeOpResult>(
      WRAP_OP(::ttnn::transformer::paged_scaled_dot_product_attention_decode), callType,
      makeTuple, device);
}

} // namespace ttnn_op_invoke
