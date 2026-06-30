// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/PagedFlashMultiLatentAttentionDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"

#include <optional>

namespace ttnn_op_invoke {

PagedFlashMultiLatentAttentionDecodeResolvedParams
resolvePagedFlashMultiLatentAttentionDecodeParams(
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &op,
    ::ttnn::MeshDevice *device) {
  PagedFlashMultiLatentAttentionDecodeResolvedParams params;

  if (op.scale.has_value()) {
    params.scale = *op.scale;
  }

  if (!op.is_causal) {
    params.programConfig = std::make_optional<
        ::ttnn::operations::transformer::SDPAProgramConfig>();
    params.programConfig->k_chunk_size = 32;
    params.programConfig->compute_with_storage_grid_size =
        device->compute_with_storage_grid_size();
  }

  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
  }

  return params;
}

template <typename Tag>
auto createPagedFlashMultiLatentAttentionDecodeTuple(
    Tag tag,
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &op,
    TensorArg query, TensorArg key, std::optional<TensorArg> value,
    TensorArg pageTable, std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink,
    const PagedFlashMultiLatentAttentionDecodeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(query, tag), resolveTensorArg(key, tag),
      value ? std::make_optional(resolveTensorArg(*value, tag)) : std::nullopt,
      op.head_dim_v, resolveTensorArg(pageTable, tag), op.is_causal,
      attentionMask ? std::make_optional(resolveTensorArg(*attentionMask, tag))
                    : std::nullopt,
      curPosTensor ? std::make_optional(resolveTensorArg(*curPosTensor, tag))
                   : std::nullopt,
      attentionSink ? std::make_optional(resolveTensorArg(*attentionSink, tag))
                    : std::nullopt,
      params.scale, /*sliding_window_size=*/std::nullopt,
      params.outputMemoryConfig, params.programConfig,
      /*compute_kernel_config=*/std::nullopt);
}

PagedFlashMultiLatentAttentionDecodeOpResult
callPagedFlashMultiLatentAttentionDecode(
    CallType callType,
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &op,
    TensorArg query, TensorArg key, std::optional<TensorArg> value,
    TensorArg pageTable, std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice *device) {
  PagedFlashMultiLatentAttentionDecodeResolvedParams params =
      resolvePagedFlashMultiLatentAttentionDecodeParams(op, device);

  auto makeTuple = [&](auto tag) {
    return createPagedFlashMultiLatentAttentionDecodeTuple(
        tag, op, query, key, value, pageTable, attentionMask, curPosTensor,
        attentionSink, params);
  };

  return callOp<PagedFlashMultiLatentAttentionDecodeOpResult>(
      WRAP_OP(::ttnn::transformer::paged_flash_multi_latent_attention_decode),
      callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
