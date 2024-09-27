// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_memory_config.h"
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
  assert(utils::isOnHost(inputTensor) or
         utils::isOnDevice(inputTensor) && "Unsupported storage type");

  assert(not utils::inSystemMemory(op->out()) &&
         "Should not be converting memory config for host tensor");

  ::ttnn::MemoryConfig memoryConfig =
      utils::createMemoryConfig(op->memcfg(), op->out());
  ::ttnn::Tensor out =
      ::ttnn::to_memory_config(inputTensor, memoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::layout
