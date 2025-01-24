// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pad.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PadOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  ::ttnn::SmallVector<int64_t> padding(op->padding()->begin(),
                                           op->padding()->end());
  std::optional<tt::tt_metal::MemoryConfig> memoryConfig =
      op->memory_config() ? std::make_optional(utils::createMemoryConfig(
                                op->memory_config(), op->out()))
                          : std::nullopt;
  float padValue = op->value();

  ::ttnn::Tensor out = ::ttnn::permute(in, padding, padValue, memoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
