// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/RotaryEmbeddingOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

RotaryEmbeddingResolvedParams resolveRotaryEmbeddingParams(
    const ::tt::target::ttnn::RotaryEmbeddingOpT &opT) {
  RotaryEmbeddingResolvedParams params;

  if (opT.token_index.has_value()) {
    params.tokenIndex = *opT.token_index;
  }

  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
  }

  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createRotaryEmbeddingTuple(
    Tag tag, const ::tt::target::ttnn::RotaryEmbeddingOpT &opT, TensorArg input,
    TensorArg cosCache, TensorArg sinCache,
    const RotaryEmbeddingResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         resolveTensorArg(cosCache, tag),
                         resolveTensorArg(sinCache, tag), params.tokenIndex,
                         params.outputMemoryConfig, params.computeConfig);
}

RotaryEmbeddingOpResult
callRotaryEmbedding(CallType callType,
                    const ::tt::target::ttnn::RotaryEmbeddingOpT &opT,
                    TensorArg input, TensorArg cosCache, TensorArg sinCache,
                    ::ttnn::MeshDevice *device) {
  RotaryEmbeddingResolvedParams params = resolveRotaryEmbeddingParams(opT);

  auto makeTuple = [&](auto tag) {
    return createRotaryEmbeddingTuple(tag, opT, input, cosCache, sinCache,
                                      params);
  };

  return callOp<RotaryEmbeddingOpResult>(
      WRAP_OP(::ttnn::experimental::rotary_embedding), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
