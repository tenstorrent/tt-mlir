// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_NLP_CONCAT_HEADS_OP_H
#define TTNN_OP_INVOKE_NLP_CONCAT_HEADS_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using NLPConcatHeadsOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct NLPConcatHeadsResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

NLPConcatHeadsResolvedParams resolveNLPConcatHeadsParams(
    const ::tt::target::ttnn::NLPConcatHeadsOpT &opT);

NLPConcatHeadsOpResult
callNLPConcatHeads(CallType callType,
                   const ::tt::target::ttnn::NLPConcatHeadsOpT &opT,
                   TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_NLP_CONCAT_HEADS_OP_H
