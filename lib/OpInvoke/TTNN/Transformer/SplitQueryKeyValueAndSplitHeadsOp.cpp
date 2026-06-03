// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/SplitQueryKeyValueAndSplitHeadsOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <tuple>
#include <variant>

namespace ttnn_op_invoke {

SplitQueryKeyValueAndSplitHeadsResolvedParams
resolveSplitQueryKeyValueAndSplitHeadsParams(
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opT) {
  SplitQueryKeyValueAndSplitHeadsResolvedParams params;
  if (opT.num_kv_heads.has_value()) {
    params.numKVHeads = *opT.num_kv_heads;
  }

  // the output memory config is passed through q_out, but both OpModel and
  // runtime always set it to 0.
  if (opT.q_out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.q_out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.q_out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createSplitQueryKeyValueAndSplitHeadsTuple(
    Tag tag, const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opT,
    TensorArg input, std::optional<TensorArg> kvInput,
    const SplitQueryKeyValueAndSplitHeadsResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag),
      kvInput ? std::make_optional(resolveTensorArg(*kvInput, tag))
              : std::nullopt,
      opT.num_heads, params.numKVHeads, opT.transpose_key,
      params.outputMemoryConfig);
}

SplitQueryKeyValueAndSplitHeadsOpResult callSplitQueryKeyValueAndSplitHeads(
    CallType callType,
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opT,
    TensorArg input, std::optional<TensorArg> kvInput,
    ::ttnn::MeshDevice *device) {
  SplitQueryKeyValueAndSplitHeadsResolvedParams params =
      resolveSplitQueryKeyValueAndSplitHeadsParams(opT);

  auto makeTuple = [&](auto tag) {
    return createSplitQueryKeyValueAndSplitHeadsTuple(tag, opT, input, kvInput,
                                                      params);
  };

  callOp(::ttnn::transformer::split_query_key_value_and_split_heads);
}

} // namespace ttnn_op_invoke
