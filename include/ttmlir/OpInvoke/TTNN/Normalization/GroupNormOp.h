// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_NORMALIZATION_GROUPNORMOP_H
#define TTMLIR_OPINVOKE_TTNN_NORMALIZATION_GROUPNORMOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm.hpp"

#include <optional>

namespace ttnn_op_invoke {

using GroupNormOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct GroupNormResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

GroupNormResolvedParams
resolveGroupNormParams(const ::tt::target::ttnn::GroupNormOpT &op,
                       CallType callType);

GroupNormOpResult
callGroupNorm(CallType callType, const ::tt::target::ttnn::GroupNormOpT &op,
              TensorArg input, std::optional<TensorArg> inputMask,
              std::optional<TensorArg> weight, std::optional<TensorArg> bias,
              ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_NORMALIZATION_GROUPNORMOP_H
