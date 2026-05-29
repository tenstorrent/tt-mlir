// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/transformer/rotaryEmbeddingOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

RotaryEmbeddingResolvedParams
resolveRotaryEmbeddingParams(const ::tt::target::ttnn::RotaryEmbeddingOpT &opT,
                             CallType callType) {
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
        operations::utils::getTensorRefMemoryConfig(*opT.out), callType);
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
                    ::ttnn::MeshDevice &targetDevice) {
  RotaryEmbeddingResolvedParams params =
      resolveRotaryEmbeddingParams(opT, callType);

  auto makeTuple = [&](auto tag) {
    return createRotaryEmbeddingTuple(tag, opT, input, cosCache, sinCache,
                                      params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::experimental::rotary_embedding,
                                      &targetDevice,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::experimental::rotary_embedding,
                                  &targetDevice,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::experimental::rotary_embedding(
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
