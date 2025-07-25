// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_memory_config.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToMemoryConfigOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in0());
  LOG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->out()),
             "Should not be converting memory config for host tensor");
  LOG_ASSERT(op->memcfg(), "ToMemoryConfigOp must have memory config");

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  ::ttnn::Tensor out =
      ::ttnn::to_memory_config(inputTensor, memoryConfig.value(), std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::layout
