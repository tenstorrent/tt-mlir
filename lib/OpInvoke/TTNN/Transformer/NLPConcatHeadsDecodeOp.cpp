// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Transformer/NLPConcatHeadsDecodeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

NLPConcatHeadsDecodeResolvedParams resolveNLPConcatHeadsDecodeParams(
    const ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &opT, CallType callType,
    TensorArg input) {
  NLPConcatHeadsDecodeResolvedParams params;

  // TODO: This branching on callType should be removed. Either add subCoreGrids
  // as a fb schema field, or remove it entirely if the op doesn't need it.
  if (callType == CallType::EXECUTE) {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);
    if (in.is_sharded() && in.shard_spec().has_value()) {
      const auto &ranges = in.shard_spec().value().grid.ranges();
      if (ranges.size() > 1 ||
          !(ranges[0].start_coord == ::tt::tt_metal::CoreCoord{0, 0})) {
        params.subCoreGrids = in.shard_spec().value().grid;
      }
    }
  } else {
    const auto &spec = std::get<::ttnn::TensorSpec>(input);
    const auto &mc = spec.tensor_layout().get_memory_config();
    if (mc.is_sharded() && mc.shard_spec().has_value()) {
      const auto &ranges = mc.shard_spec().value().grid.ranges();
      if (ranges.size() > 1 ||
          !(ranges[0].start_coord == ::tt::tt_metal::CoreCoord{0, 0})) {
        params.subCoreGrids = mc.shard_spec().value().grid;
      }
    }
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
auto createNLPConcatHeadsDecodeTuple(
    Tag tag, const ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &opT,
    TensorArg input, const NLPConcatHeadsDecodeResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), opT.num_heads, params.outputMemoryConfig,
      /*optional_output_tensor=*/std::nullopt, params.subCoreGrids);
}

NLPConcatHeadsDecodeOpResult
callNLPConcatHeadsDecode(CallType callType,
                         const ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &opT,
                         TensorArg input, ::ttnn::MeshDevice *device) {
  NLPConcatHeadsDecodeResolvedParams params =
      resolveNLPConcatHeadsDecodeParams(opT, callType, input);

  auto makeTuple = [&](auto tag) {
    return createNLPConcatHeadsDecodeTuple(tag, opT, input, params);
  };

  return callOp<NLPConcatHeadsDecodeOpResult>(
      WRAP_OP(::ttnn::experimental::nlp_concat_heads_decode), callType,
      makeTuple, device);
}

} // namespace ttnn_op_invoke
