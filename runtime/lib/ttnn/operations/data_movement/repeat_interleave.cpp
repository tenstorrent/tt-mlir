// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/repeat_interleave.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::RepeatInterleaveOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  DEBUG_ASSERT(input.is_allocated());

  uint32_t repeats = op->repeats();
  int32_t dim = op->dim();
  std::optional<tt::tt_metal::MemoryConfig> memoryConfig =
      op->memory_config() ? std::make_optional(utils::createMemoryConfig(
                                op->memory_config(), op->out()))
                          : std::nullopt;

  ::ttnn::Tensor out =
      ::ttnn::repeat_interleave(input, repeats, dim, memoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
