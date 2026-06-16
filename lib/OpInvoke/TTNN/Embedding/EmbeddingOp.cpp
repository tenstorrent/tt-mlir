// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Embedding/EmbeddingOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include <optional>

namespace ttnn_op_invoke {

EmbeddingResolvedParams
resolveEmbeddingParams(const ::tt::target::ttnn::EmbeddingOpT &op) {
  EmbeddingResolvedParams params;

  bool isTiled = op.out && op.out->desc && op.out->desc->layout &&
                 op.out->desc->layout->memory_desc &&
                 op.out->desc->layout->memory_desc->tile_shape;
  params.layout = isTiled ? ::ttnn::TILE_LAYOUT : ::ttnn::ROW_MAJOR_LAYOUT;

  params.embeddingsType = ::ttnn::prim::EmbeddingsType::GENERIC;

  if (op.out) {
    params.dtype = operations::utils::getDataType(*op.out);
  }

  if (op.out) {
    params.memoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.memoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEmbeddingTuple(Tag tag, TensorArg input, TensorArg weight,
                          const EmbeddingResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag),
      /*pad_token=*/std::nullopt, params.layout, params.embeddingsType,
      params.dtype, params.memoryConfig);
}

EmbeddingOpResult callEmbedding(CallType callType,
                                const ::tt::target::ttnn::EmbeddingOpT &op,
                                TensorArg input, TensorArg weight,
                                ::ttnn::MeshDevice *device) {
  EmbeddingResolvedParams params = resolveEmbeddingParams(op);

  auto makeTuple = [&](auto tag) {
    return createEmbeddingTuple(tag, input, weight, params);
  };

  return callOp<EmbeddingOpResult>(WRAP_OP(::ttnn::embedding), callType,
                                   makeTuple, device);
}

} // namespace ttnn_op_invoke
