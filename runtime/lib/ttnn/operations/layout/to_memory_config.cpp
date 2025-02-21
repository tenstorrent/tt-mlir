// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_memory_config.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {

// TODO(bug #272): right now hardcoding tilize/untilize, should determine with
// tile shape blocked by issue #272
void run(const ::tt::target::ttnn::ToMemoryConfigOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in0()->global_id());
  DEBUG_ASSERT(inputTensor.is_allocated());
  LOG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->out()),
             "Should not be converting memory config for host tensor");
  LOG_ASSERT(op->memcfg(), "ToMemoryConfigOp must have memory config");
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  ::ttnn::Tensor out =
      ::ttnn::to_memory_config(inputTensor, memoryConfig.value(), std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::layout
