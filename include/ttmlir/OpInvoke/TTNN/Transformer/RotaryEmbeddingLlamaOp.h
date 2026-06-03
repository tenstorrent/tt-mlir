// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_ROTARY_EMBEDDING_LLAMA_OP_H
#define TTNN_OP_INVOKE_ROTARY_EMBEDDING_LLAMA_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using RotaryEmbeddingLlamaOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct RotaryEmbeddingLlamaResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
};

RotaryEmbeddingLlamaResolvedParams resolveRotaryEmbeddingLlamaParams(
    const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opT);

RotaryEmbeddingLlamaOpResult callRotaryEmbeddingLlama(
    CallType callType, const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opT,
    TensorArg input, TensorArg cosCache, TensorArg sinCache, TensorArg transMat,
    ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_ROTARY_EMBEDDING_LLAMA_OP_H
