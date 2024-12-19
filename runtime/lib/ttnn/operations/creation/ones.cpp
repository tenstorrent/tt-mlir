// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/ones.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

#include <functional>
#include <optional>
#include <variant>

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::OnesOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Shape shape = ::ttnn::Shape(::tt::tt_metal::LegacyShape(
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(*op->shape())));

  std::optional<::ttnn::DataType> dtype = std::optional<::ttnn::DataType>();
  std::optional<::ttnn::Layout> layout = std::optional<::ttnn::Layout>();
  std::optional<std::reference_wrapper<::ttnn::Device>> device = std::nullopt;
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      std::optional<::ttnn::MemoryConfig>();

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  if (op->layout()) {
    layout = ::tt::runtime::ttnn::utils::toTTNNLayout(*(op->layout()));
  }

  if (op->device()) {
    DeviceVariant targetDevice =
        context.getTargetDevice(op->device()->global_id());
    LOG_ASSERT(std::holds_alternative<std::reference_wrapper<::ttnn::Device>>(
                   targetDevice),
               "ttnn::ones does not support MeshDevice.");
    device = std::get<std::reference_wrapper<::ttnn::Device>>(targetDevice);
  }

  if (op->memcfg()) {
    memoryConfig = utils::createMemoryConfig(op->memcfg(), op->out());
  }

  ::ttnn::Tensor out = ::ttnn::ones(shape, dtype, layout, device, memoryConfig);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
