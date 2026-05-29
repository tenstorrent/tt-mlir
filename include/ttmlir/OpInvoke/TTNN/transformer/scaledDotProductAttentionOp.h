// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_SCALED_DOT_PRODUCT_ATTENTION_OP_H
#define TTNN_OP_INVOKE_SCALED_DOT_PRODUCT_ATTENTION_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ScaledDotProductAttentionOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ScaledDotProductAttentionResolvedParams {
  std::optional<float> scale;
  std::optional<uint32_t> slidingWindowSize;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ScaledDotProductAttentionResolvedParams resolveScaledDotProductAttentionParams(
    const ::tt::target::ttnn::ScaledDotProductAttentionOpT &opT,
    CallType callType);

ScaledDotProductAttentionOpResult callScaledDotProductAttention(
    CallType callType,
    const ::tt::target::ttnn::ScaledDotProductAttentionOpT &opT,
    TensorArg query, TensorArg key, TensorArg value,
    std::optional<TensorArg> attentionMask,
    std::optional<TensorArg> attentionSink, ::ttnn::MeshDevice &targetDevice);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_SCALED_DOT_PRODUCT_ATTENTION_OP_H
