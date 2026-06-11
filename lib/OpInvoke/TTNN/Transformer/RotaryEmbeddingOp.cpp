// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/RotaryEmbeddingOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"

#include <optional>

namespace ttnn_op_invoke {

RotaryEmbeddingResolvedParams
resolveRotaryEmbeddingParams(const ::tt::target::ttnn::RotaryEmbeddingOpT &op) {
  RotaryEmbeddingResolvedParams params;

  if (op.token_index.has_value()) {
    params.tokenIndex = *op.token_index;
  }

  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }

  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createRotaryEmbeddingTuple(
    Tag tag, const ::tt::target::ttnn::RotaryEmbeddingOpT &op, TensorArg input,
    TensorArg cosCache, TensorArg sinCache,
    const RotaryEmbeddingResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         resolveTensorArg(cosCache, tag),
                         resolveTensorArg(sinCache, tag), params.tokenIndex,
                         params.outputMemoryConfig, params.computeConfig);
}

RotaryEmbeddingOpResult
callRotaryEmbedding(CallType callType,
                    const ::tt::target::ttnn::RotaryEmbeddingOpT &op,
                    TensorArg input, TensorArg cosCache, TensorArg sinCache,
                    ::ttnn::MeshDevice *device) {
  RotaryEmbeddingResolvedParams params = resolveRotaryEmbeddingParams(op);

  auto makeTuple = [&](auto tag) {
    return createRotaryEmbeddingTuple(tag, op, input, cosCache, sinCache,
                                      params);
  };

  return callOp<RotaryEmbeddingOpResult>(
      WRAP_OP(::ttnn::experimental::rotary_embedding), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
