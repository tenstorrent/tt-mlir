// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_LAYER_NORM_OP_H
#define TTNN_OP_INVOKE_LAYER_NORM_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using LayerNormOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct LayerNormResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

LayerNormResolvedParams
resolveLayerNormParams(const ::tt::target::ttnn::LayerNormOpT &opT);

LayerNormOpResult
callLayerNorm(CallType callType, const ::tt::target::ttnn::LayerNormOpT &opT,
              TensorArg input, std::optional<TensorArg> weight,
              std::optional<TensorArg> bias, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_LAYER_NORM_OP_H
