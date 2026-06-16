// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_EMBEDDING_EMBEDDINGBACKWARDOP_H
#define TTMLIR_OPINVOKE_TTNN_EMBEDDING_EMBEDDINGBACKWARDOP_H

#include "operations/embedding_backward/embedding_backward.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using EmbeddingBackwardOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EmbeddingBackwardResolvedParams {
  std::optional<::ttnn::DataType> dtype;
  std::optional<::ttnn::MemoryConfig> memoryConfig;
};

EmbeddingBackwardResolvedParams resolveEmbeddingBackwardParams(
    const ::tt::target::ttnn::EmbeddingBackwardOpT &op);

EmbeddingBackwardOpResult
callEmbeddingBackward(CallType callType,
                      const ::tt::target::ttnn::EmbeddingBackwardOpT &op,
                      TensorArg input, TensorArg weight, TensorArg inGradient,
                      ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_EMBEDDING_EMBEDDINGBACKWARDOP_H
