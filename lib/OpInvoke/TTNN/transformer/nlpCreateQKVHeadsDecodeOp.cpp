// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/transformer/nlpCreateQKVHeadsDecodeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>

namespace ttnn_op_invoke {

NLPCreateQKVHeadsDecodeResolvedParams resolveNLPCreateQKVHeadsDecodeParams(
    const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &opT,
    CallType callType) {
  NLPCreateQKVHeadsDecodeResolvedParams params;
  params.optionalOutputTensors = std::nullopt;

  if (opT.memcfg) {
    params.outputMemoryConfig =
        operations::utils::createMemoryConfigIfNeeded(*opT.memcfg, callType);
  }

  return params;
}

template <typename Tag>
auto createNLPCreateQKVHeadsDecodeTuple(
    Tag tag, const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &opT,
    TensorArg input, std::optional<TensorArg> batchOffset,
    const NLPCreateQKVHeadsDecodeResolvedParams &params) {
  std::optional<const uint32_t> numKVHeads =
      opT.num_kv_heads.has_value()
          ? std::optional<const uint32_t>(*opT.num_kv_heads)
          : std::nullopt;
  std::optional<const bool> overlapQKCoregrid =
      opT.overlap_qk_coregrid.has_value()
          ? std::optional<const bool>(*opT.overlap_qk_coregrid)
          : std::nullopt;
  std::optional<const uint32_t> sliceSize =
      opT.slice_size.has_value()
          ? std::optional<const uint32_t>(*opT.slice_size)
          : std::nullopt;
  return std::make_tuple(
      resolveTensorArg(input, tag), opT.num_heads, numKVHeads,
      params.optionalOutputTensors, overlapQKCoregrid,
      batchOffset ? std::make_optional(resolveTensorArg(*batchOffset, tag))
                  : std::nullopt,
      sliceSize, params.outputMemoryConfig);
}

NLPCreateQKVHeadsDecodeOpResult callNLPCreateQKVHeadsDecode(
    CallType callType,
    const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &opT, TensorArg input,
    std::optional<TensorArg> batchOffset, ::ttnn::MeshDevice &targetDevice) {
  NLPCreateQKVHeadsDecodeResolvedParams params =
      resolveNLPCreateQKVHeadsDecodeParams(opT, callType);

  auto makeTuple = [&](auto tag) {
    return createNLPCreateQKVHeadsDecodeTuple(tag, opT, input, batchOffset,
                                              params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(
              ::ttnn::experimental::nlp_create_qkv_heads_decode, &targetDevice,
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(
              ::ttnn::experimental::nlp_create_qkv_heads_decode, &targetDevice,
              std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE: {
    auto executeTuple = makeTuple(ExecuteTag{});
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::experimental::nlp_create_qkv_heads_decode(
              std::forward<decltype(args)>(args)...);
        },
        executeTuple);
  }
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
