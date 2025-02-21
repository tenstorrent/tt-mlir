// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/arange.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/types.hpp"

#include <functional>
#include <variant>

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::ArangeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::DataType dtype =
      ::ttnn::DataType::BFLOAT16; // Default in arange implementation
  std::optional<std::reference_wrapper<::ttnn::IDevice>> device = std::nullopt;
  ::ttnn::MemoryConfig memoryConfig =
      ::ttnn::DRAM_MEMORY_CONFIG; // Default in arange implementation

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  if (op->memcfg()) {
    memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg())
            .value();
  }

  if (op->device()) {
    // ttnn::arange supports no device (host) and single device
    DeviceVariant targetDevice =
        context.getTargetDevice(op->device()->global_id());

    LOG_ASSERT(std::holds_alternative<std::reference_wrapper<::ttnn::IDevice>>(
                   targetDevice),
               "ttnn::arange does not support MeshDevice.");
    device = std::make_optional(
        std::get<std::reference_wrapper<::ttnn::IDevice>>(targetDevice));
  }
  ::ttnn::Tensor out = ::ttnn::arange(op->start(), op->end(), op->step(), dtype,
                                      device, memoryConfig);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
