// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_NLP_CONCAT_HEADS_DECODE_OP_H
#define TTNN_OP_INVOKE_NLP_CONCAT_HEADS_DECODE_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using NLPConcatHeadsDecodeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct NLPConcatHeadsDecodeResolvedParams {
  std::optional<::tt::tt_metal::CoreRangeSet> subCoreGrids;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

NLPConcatHeadsDecodeResolvedParams resolveNLPConcatHeadsDecodeParams(
    const ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &op, CallType callType,
    TensorArg input);

NLPConcatHeadsDecodeOpResult
callNLPConcatHeadsDecode(CallType callType,
                         const ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &op,
                         TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_NLP_CONCAT_HEADS_DECODE_OP_H
