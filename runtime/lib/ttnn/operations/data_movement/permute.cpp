// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/permute.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"

#include <vector>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PermuteOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  std::vector<int64_t> permutation(op->permutation()->begin(),
                                   op->permutation()->end());
  std::optional<tt::tt_metal::MemoryConfig> memoryConfig =
      op->memory_config() ? std::make_optional(utils::createMemoryConfig(
                                op->memory_config(), op->out()))
                          : std::nullopt;
  float padValue = op->pad_value();

  ::ttnn::Tensor out = ::ttnn::permute(in, permutation, memoryConfig, padValue);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
