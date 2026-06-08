// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/NLPConcatHeadsOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

NLPConcatHeadsResolvedParams
resolveNLPConcatHeadsParams(const ::tt::target::ttnn::NLPConcatHeadsOpT &op) {
  NLPConcatHeadsResolvedParams params;
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
auto createNLPConcatHeadsTuple(
    Tag tag, const ::tt::target::ttnn::NLPConcatHeadsOpT & /*op*/,
    TensorArg input, const NLPConcatHeadsResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig);
}

NLPConcatHeadsOpResult
callNLPConcatHeads(CallType callType,
                   const ::tt::target::ttnn::NLPConcatHeadsOpT &op,
                   TensorArg input, ::ttnn::MeshDevice *device) {
  NLPConcatHeadsResolvedParams params = resolveNLPConcatHeadsParams(op);

  auto makeTuple = [&](auto tag) {
    return createNLPConcatHeadsTuple(tag, op, input, params);
  };

  return callOp<NLPConcatHeadsOpResult>(
      WRAP_OP(::ttnn::experimental::nlp_concat_heads), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
