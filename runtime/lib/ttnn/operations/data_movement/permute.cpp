// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttnn/common/constants.hpp"

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

  // Has to called with the verbose invoke, since calling it with composite=true
  // results in an error with the message "Only 4D tensor are supported for
  // permute."
  ::ttnn::Tensor out =
      ::ttnn::permute(/*queue_id=*/::ttnn::DefaultQueueId, /*input_tensor=*/in,
                      /*dims=*/permutation, /*memory_config=*/memoryConfig,
                      /*composite=*/false, /*pad_value=*/padValue);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
