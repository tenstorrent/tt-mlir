// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Embedding/EmbeddingBackwardOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

EmbeddingBackwardResolvedParams resolveEmbeddingBackwardParams(
    const ::tt::target::ttnn::EmbeddingBackwardOpT &op) {
  EmbeddingBackwardResolvedParams params;

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
auto createEmbeddingBackwardTuple(
    Tag tag, const ::tt::target::ttnn::EmbeddingBackwardOpT &op,
    TensorArg input, TensorArg weight, TensorArg inGradient,
    const EmbeddingBackwardResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag),
      resolveTensorArg(inGradient, tag), params.dtype, params.memoryConfig);
}

EmbeddingBackwardOpResult
callEmbeddingBackward(CallType callType,
                      const ::tt::target::ttnn::EmbeddingBackwardOpT &op,
                      TensorArg input, TensorArg weight, TensorArg inGradient,
                      ::ttnn::MeshDevice *device) {
  EmbeddingBackwardResolvedParams params = resolveEmbeddingBackwardParams(op);

  auto makeTuple = [&](auto tag) {
    return createEmbeddingBackwardTuple(tag, op, input, weight, inGradient,
                                        params);
  };

  return callOp<EmbeddingBackwardOpResult>(WRAP_OP(::ttnn::embedding_bw),
                                           callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
