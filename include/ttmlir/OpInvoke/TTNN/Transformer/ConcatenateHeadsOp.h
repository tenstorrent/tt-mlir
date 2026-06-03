// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_CONCATENATE_HEADS_OP_H
#define TTNN_OP_INVOKE_CONCATENATE_HEADS_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ConcatenateHeadsOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ConcatenateHeadsResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ConcatenateHeadsResolvedParams resolveConcatenateHeadsParams(
    const ::tt::target::ttnn::ConcatenateHeadsOpT &opT);

ConcatenateHeadsOpResult
callConcatenateHeads(CallType callType,
                     const ::tt::target::ttnn::ConcatenateHeadsOpT &opT,
                     TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_CONCATENATE_HEADS_OP_H
