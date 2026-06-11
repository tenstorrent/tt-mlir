// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_TRANSFORMER_CONCATENATEHEADSOP_H
#define TTMLIR_OPINVOKE_TTNN_TRANSFORMER_CONCATENATEHEADSOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ConcatenateHeadsOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ConcatenateHeadsResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ConcatenateHeadsResolvedParams resolveConcatenateHeadsParams(
    const ::tt::target::ttnn::ConcatenateHeadsOpT &op);

ConcatenateHeadsOpResult
callConcatenateHeads(CallType callType,
                     const ::tt::target::ttnn::ConcatenateHeadsOpT &op,
                     TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_TRANSFORMER_CONCATENATEHEADSOP_H
