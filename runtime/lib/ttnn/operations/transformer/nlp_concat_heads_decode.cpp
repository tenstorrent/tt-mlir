// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::NLPConcatHeadsDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  // Pass in sub_core_grids when metal would infer it is required
  std::optional<::tt::tt_metal::CoreRangeSet> subCoreGrids = std::nullopt;
  if (in.is_sharded() && in.shard_spec().has_value()) {
    const auto &inputCoreRanges = in.shard_spec().value().grid.ranges();
    if (inputCoreRanges.size() > 1 ||
        !(inputCoreRanges[0].start_coord == ::tt::tt_metal::CoreCoord{0, 0})) {
      subCoreGrids = in.shard_spec().value().grid;
    }
  }

  ::ttnn::Tensor out = ::ttnn::experimental::nlp_concat_heads_decode(
      in, op->num_heads(), outputMemoryConfig, std::nullopt, subCoreGrids);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::transformer
