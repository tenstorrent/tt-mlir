// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/NlpConcatHeadsOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

NLPConcatHeadsResolvedParams
resolveNLPConcatHeadsParams(const ::tt::target::ttnn::NLPConcatHeadsOpT &opT) {
  NLPConcatHeadsResolvedParams params;
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
auto createNLPConcatHeadsTuple(
    Tag tag, const ::tt::target::ttnn::NLPConcatHeadsOpT & /*opT*/,
    TensorArg input, const NLPConcatHeadsResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig);
}

NLPConcatHeadsOpResult
callNLPConcatHeads(CallType callType,
                   const ::tt::target::ttnn::NLPConcatHeadsOpT &opT,
                   TensorArg input, ::ttnn::MeshDevice *device) {
  NLPConcatHeadsResolvedParams params = resolveNLPConcatHeadsParams(opT);

  auto makeTuple = [&](auto tag) {
    return createNLPConcatHeadsTuple(tag, opT, input, params);
  };

  callOp(::ttnn::experimental::nlp_concat_heads);
}

} // namespace ttnn_op_invoke
