// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_TRANSFORMER_NLPCREATEQKVHEADSDECODEOP_H
#define TTMLIR_OPINVOKE_TTNN_TRANSFORMER_NLPCREATEQKVHEADSDECODEOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.hpp"

#include <optional>
#include <tuple>

namespace ttnn_op_invoke {

using NLPCreateQKVHeadsDecodeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse,
                 std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor>>;

struct NLPCreateQKVHeadsDecodeResolvedParams {
  std::optional<std::array<::ttnn::Tensor, 3>> optionalOutputTensors;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

NLPCreateQKVHeadsDecodeResolvedParams resolveNLPCreateQKVHeadsDecodeParams(
    const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &op);

NLPCreateQKVHeadsDecodeOpResult callNLPCreateQKVHeadsDecode(
    CallType callType, const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &op,
    TensorArg input, std::optional<TensorArg> batchOffset,
    ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_TRANSFORMER_NLPCREATEQKVHEADSDECODEOP_H
