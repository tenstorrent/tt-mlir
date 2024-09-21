// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::FullOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap devicePool = context.devicePool;
  ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  auto shape = ::ttnn::Shape(::tt::tt_metal::LegacyShape(
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->out()->desc()->shape())));
  float fillValue = op->fill_value();
  // TODO(bug #272), determine correct layout by tile shape in the future
  ::ttnn::Layout outputLayout = ::ttnn::Layout::ROW_MAJOR;
  std::optional<std::reference_wrapper<::ttnn::Device>> outputDevice =
      std::make_optional(std::ref(device));
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      std::make_optional(utils::createMemoryConfig(op->out()));

  ::ttnn::Tensor out =
      ::ttnn::full(shape, fillValue, outputDataType, outputLayout, outputDevice,
                   outputMemoryConfig);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
