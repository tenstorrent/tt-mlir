// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_device.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {

void run(const ::tt::target::ttnn::ToDeviceOp *op, ProgramContext &context) {
  LOG_ASSERT(op->device(), "ToDeviceOp must have a device");
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  DEBUG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::to_device on a device tensor");

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ::ttnn::Tensor out =
      ::ttnn::to_device(inputTensor, &targetDevice, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::layout
