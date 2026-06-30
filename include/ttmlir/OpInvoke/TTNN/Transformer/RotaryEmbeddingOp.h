// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_TRANSFORMER_ROTARYEMBEDDINGOP_H
#define TTMLIR_OPINVOKE_TTNN_TRANSFORMER_ROTARYEMBEDDINGOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"

#include <optional>

namespace ttnn_op_invoke {

using RotaryEmbeddingOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct RotaryEmbeddingResolvedParams {
  std::optional<uint32_t> tokenIndex;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
};

RotaryEmbeddingResolvedParams
resolveRotaryEmbeddingParams(const ::tt::target::ttnn::RotaryEmbeddingOpT &op);

RotaryEmbeddingOpResult
callRotaryEmbedding(CallType callType,
                    const ::tt::target::ttnn::RotaryEmbeddingOpT &op,
                    TensorArg input, TensorArg cosCache, TensorArg sinCache,
                    ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_TRANSFORMER_ROTARYEMBEDDINGOP_H
