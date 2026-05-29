// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/transformer/pagedFlashMultiLatentAttentionDecodeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

PagedFlashMultiLatentAttentionDecodeResolvedParams
resolvePagedFlashMultiLatentAttentionDecodeParams(
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &opT,
    CallType callType, ::ttnn::MeshDevice &targetDevice) {
  PagedFlashMultiLatentAttentionDecodeResolvedParams params;

  if (opT.scale.has_value()) {
    params.scale = *opT.scale;
  }

  if (!opT.is_causal) {
    params.programConfig = std::make_optional<
        ::ttnn::operations::transformer::SDPAProgramConfig>();
    params.programConfig->k_chunk_size = 32; // Required for non-causal
    params.programConfig->compute_with_storage_grid_size =
        targetDevice.compute_with_storage_grid_size();
  }

  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out), callType);
  }

  return params;
}

template <typename Tag>
auto createPagedFlashMultiLatentAttentionDecodeTuple(
    Tag tag,
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, std::optional<TensorArg> value,
    TensorArg pageTable, std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice &targetDevice,
    const PagedFlashMultiLatentAttentionDecodeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(query, tag), resolveTensorArg(key, tag),
      value ? std::make_optional(resolveTensorArg(*value, tag)) : std::nullopt,
      opT.head_dim_v, resolveTensorArg(pageTable, tag), opT.is_causal,
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
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, std::optional<TensorArg> value,
    TensorArg pageTable, std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice &targetDevice) {
  PagedFlashMultiLatentAttentionDecodeResolvedParams params =
      resolvePagedFlashMultiLatentAttentionDecodeParams(opT, callType,
                                                        targetDevice);

  auto makeTuple = [&](auto tag) {
    return createPagedFlashMultiLatentAttentionDecodeTuple(
        tag, opT, query, key, value, pageTable, attentionMask, curPosTensor,
        attentionSink, targetDevice, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(
              ::ttnn::transformer::paged_flash_multi_latent_attention_decode,
              &targetDevice, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(
              ::ttnn::transformer::paged_flash_multi_latent_attention_decode,
              &targetDevice, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::transformer::paged_flash_multi_latent_attention_decode(
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
