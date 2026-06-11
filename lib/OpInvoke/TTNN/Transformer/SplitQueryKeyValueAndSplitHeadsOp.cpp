// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/SplitQueryKeyValueAndSplitHeadsOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"

#include <optional>
#include <tuple>

namespace ttnn_op_invoke {

SplitQueryKeyValueAndSplitHeadsResolvedParams
resolveSplitQueryKeyValueAndSplitHeadsParams(
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &op) {
  SplitQueryKeyValueAndSplitHeadsResolvedParams params;
  if (op.num_kv_heads.has_value()) {
    params.numKVHeads = *op.num_kv_heads;
  }

  // The output memory config is passed through q_out
  if (op.q_out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.q_out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.q_out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createSplitQueryKeyValueAndSplitHeadsTuple(
    Tag tag, const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &op,
    TensorArg input, std::optional<TensorArg> kvInput,
    const SplitQueryKeyValueAndSplitHeadsResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag),
      kvInput ? std::make_optional(resolveTensorArg(*kvInput, tag))
              : std::nullopt,
      op.num_heads, params.numKVHeads, op.transpose_key,
      params.outputMemoryConfig);
}

SplitQueryKeyValueAndSplitHeadsOpResult callSplitQueryKeyValueAndSplitHeads(
    CallType callType,
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &op,
    TensorArg input, std::optional<TensorArg> kvInput,
    ::ttnn::MeshDevice *device) {
  SplitQueryKeyValueAndSplitHeadsResolvedParams params =
      resolveSplitQueryKeyValueAndSplitHeadsParams(op);

  auto makeTuple = [&](auto tag) {
    return createSplitQueryKeyValueAndSplitHeadsTuple(tag, op, input, kvInput,
                                                      params);
  };

  return callOp<SplitQueryKeyValueAndSplitHeadsOpResult>(
      WRAP_OP(::ttnn::transformer::split_query_key_value_and_split_heads),
      callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
