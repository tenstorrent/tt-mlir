// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_EMBEDDING_EMBEDDINGOP_H
#define TTMLIR_OPINVOKE_TTNN_EMBEDDING_EMBEDDINGOP_H

#include "operations/embedding/embedding.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using EmbeddingOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EmbeddingResolvedParams {
  std::optional<::ttnn::Layout> layout;
  ::ttnn::prim::EmbeddingsType embeddingsType;
  std::optional<::ttnn::DataType> dtype;
  std::optional<::ttnn::MemoryConfig> memoryConfig;
};

EmbeddingResolvedParams
resolveEmbeddingParams(const ::tt::target::ttnn::EmbeddingOpT &op);

EmbeddingOpResult callEmbedding(CallType callType,
                                const ::tt::target::ttnn::EmbeddingOpT &op,
                                TensorArg input, TensorArg weight,
                                ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_EMBEDDING_EMBEDDINGOP_H
