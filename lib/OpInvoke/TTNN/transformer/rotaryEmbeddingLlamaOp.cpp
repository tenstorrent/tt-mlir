// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/transformer/rotaryEmbeddingLlamaOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

RotaryEmbeddingLlamaResolvedParams resolveRotaryEmbeddingLlamaParams(
    const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opT, CallType callType) {
  RotaryEmbeddingLlamaResolvedParams params;

  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
  }

  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createRotaryEmbeddingLlamaTuple(
    Tag tag, const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opT,
    TensorArg input, TensorArg cosCache, TensorArg sinCache, TensorArg transMat,
    const RotaryEmbeddingLlamaResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(cosCache, tag),
      resolveTensorArg(sinCache, tag), resolveTensorArg(transMat, tag),
      opT.is_decode_mode, params.outputMemoryConfig, params.computeConfig);
}

RotaryEmbeddingLlamaOpResult callRotaryEmbeddingLlama(
    CallType callType, const ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opT,
    TensorArg input, TensorArg cosCache, TensorArg sinCache, TensorArg transMat,
    ::ttnn::MeshDevice &targetDevice) {
  RotaryEmbeddingLlamaResolvedParams params =
      resolveRotaryEmbeddingLlamaParams(opT, callType);

  auto makeTuple = [&](auto tag) {
    return createRotaryEmbeddingLlamaTuple(tag, opT, input, cosCache, sinCache,
                                           transMat, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(
              ::ttnn::experimental::rotary_embedding_llama, &targetDevice,
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::experimental::rotary_embedding_llama,
                                  &targetDevice,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::experimental::rotary_embedding_llama(
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
