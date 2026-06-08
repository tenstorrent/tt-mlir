// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/ConcatenateHeadsOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

ConcatenateHeadsResolvedParams resolveConcatenateHeadsParams(
    const ::tt::target::ttnn::ConcatenateHeadsOpT &opT) {
  ConcatenateHeadsResolvedParams params;
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
auto createConcatenateHeadsTuple(
    Tag tag, const ::tt::target::ttnn::ConcatenateHeadsOpT & /*opT*/,
    TensorArg input, const ConcatenateHeadsResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig);
}

ConcatenateHeadsOpResult
callConcatenateHeads(CallType callType,
                     const ::tt::target::ttnn::ConcatenateHeadsOpT &opT,
                     TensorArg input, ::ttnn::MeshDevice *device) {
  ConcatenateHeadsResolvedParams params = resolveConcatenateHeadsParams(opT);

  auto makeTuple = [&](auto tag) {
    return createConcatenateHeadsTuple(tag, opT, input, params);
  };

  return callOp<ConcatenateHeadsOpResult>(
      ::ttnn::transformer::concatenate_heads, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
