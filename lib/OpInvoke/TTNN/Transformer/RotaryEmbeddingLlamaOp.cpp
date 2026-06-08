// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/RotaryEmbeddingLlamaOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

RotaryEmbeddingLlamaResolvedParams resolveRotaryEmbeddingLlamaParams(
    const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &op) {
  RotaryEmbeddingLlamaResolvedParams params;

  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }

  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createRotaryEmbeddingLlamaTuple(
    Tag tag, const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &op,
    TensorArg input, TensorArg cosCache, TensorArg sinCache, TensorArg transMat,
    const RotaryEmbeddingLlamaResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(cosCache, tag),
      resolveTensorArg(sinCache, tag), resolveTensorArg(transMat, tag),
      op.is_decode_mode, params.outputMemoryConfig, params.computeConfig);
}

RotaryEmbeddingLlamaOpResult callRotaryEmbeddingLlama(
    CallType callType, const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &op,
    TensorArg input, TensorArg cosCache, TensorArg sinCache, TensorArg transMat,
    ::ttnn::MeshDevice *device) {
  RotaryEmbeddingLlamaResolvedParams params =
      resolveRotaryEmbeddingLlamaParams(op);

  auto makeTuple = [&](auto tag) {
    return createRotaryEmbeddingLlamaTuple(tag, op, input, cosCache, sinCache,
                                           transMat, params);
  };

  return callOp<RotaryEmbeddingLlamaOpResult>(
      WRAP_OP(::ttnn::experimental::rotary_embedding_llama), callType,
      makeTuple, device);
}

} // namespace ttnn_op_invoke
