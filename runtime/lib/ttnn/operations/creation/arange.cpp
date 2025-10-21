// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/arange.h"
#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/types.hpp"

#include <functional>
#include <variant>

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::ArangeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::DataType dtype =
      ::ttnn::DataType::BFLOAT16; // Default in arange implementation
  ::ttnn::Layout layout = ::ttnn::Layout::ROW_MAJOR;
  OptionalMeshDeviceRef targetDevice = std::nullopt;
  ::ttnn::MemoryConfig memoryConfig =
      ::ttnn::DRAM_MEMORY_CONFIG; // Default in arange implementation

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  if (op->layout()) {
    layout = ::tt::runtime::ttnn::utils::toTTNNLayout(*(op->layout()));
  }

  if (op->memcfg()) {
    memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg())
            .value();
  }

  if (op->device()) {
    targetDevice = std::ref(context.getMeshDevice());
  }

  ::ttnn::Tensor out = ::ttnn::arange(op->start(), op->end(), op->step(), dtype,
                                      targetDevice, memoryConfig, layout);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
