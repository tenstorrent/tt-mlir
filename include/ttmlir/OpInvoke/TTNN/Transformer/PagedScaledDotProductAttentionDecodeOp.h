// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_TRANSFORMER_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEOP_H
#define TTMLIR_OPINVOKE_TTNN_TRANSFORMER_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"

#include <optional>

namespace ttnn_op_invoke {

using PagedScaledDotProductAttentionDecodeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PagedScaledDotProductAttentionDecodeResolvedParams {
  std::optional<float> scale;
  std::optional<uint32_t> slidingWindowSize;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig;
};

PagedScaledDotProductAttentionDecodeResolvedParams
resolvePagedScaledDotProductAttentionDecodeParams(
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &op,
    ::ttnn::MeshDevice *device);

PagedScaledDotProductAttentionDecodeOpResult
callPagedScaledDotProductAttentionDecode(
    CallType callType,
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &op,
    TensorArg query, TensorArg key, TensorArg value, TensorArg pageTable,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> curPosTensor,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_TRANSFORMER_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEOP_H
