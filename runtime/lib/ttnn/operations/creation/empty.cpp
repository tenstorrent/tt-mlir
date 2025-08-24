// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/empty.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Shape shape = ::tt::runtime::ttnn::operations::utils::toTTNNShape(
      *op->out()->desc()->shape());
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());

  ::ttnn::Tensor out;
  if (op->device()) {
    std::optional<::ttnn::MemoryConfig> memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
    LOG_ASSERT(memoryConfig.has_value(),
               "Memory config is required for device tensors");
    ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
    out =
        ::ttnn::empty(shape, dtype, layout, &meshDevice, memoryConfig.value());
  } else {
    out = ::ttnn::zeros(shape, dtype, layout);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
