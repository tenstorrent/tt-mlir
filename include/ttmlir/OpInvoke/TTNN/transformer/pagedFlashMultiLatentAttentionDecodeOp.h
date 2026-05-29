// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_PAGED_FLASH_MULTI_LATENT_ATTENTION_DECODE_OP_H
#define TTNN_OP_INVOKE_PAGED_FLASH_MULTI_LATENT_ATTENTION_DECODE_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using PagedFlashMultiLatentAttentionDecodeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PagedFlashMultiLatentAttentionDecodeResolvedParams {
  std::optional<float> scale;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig;
};

PagedFlashMultiLatentAttentionDecodeResolvedParams
resolvePagedFlashMultiLatentAttentionDecodeParams(
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &opT,
    CallType callType, ::ttnn::MeshDevice &targetDevice);

PagedFlashMultiLatentAttentionDecodeOpResult
callPagedFlashMultiLatentAttentionDecode(
    CallType callType,
    const ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &opT,
    TensorArg query, TensorArg key, std::optional<TensorArg> value,
    TensorArg pageTable, std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice &targetDevice);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_PAGED_FLASH_MULTI_LATENT_ATTENTION_DECODE_OP_H
