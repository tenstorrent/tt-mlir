// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::FullOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  auto shape = ::ttnn::Shape(::tt::tt_metal::LegacyShape(
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->out()->desc()->shape())));
  float fillValue = op->fill_value();

  // TODO(bug #272), determine correct layout by tile shape in the future
  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  ::ttnn::Layout outputLayout = ::ttnn::Layout::ROW_MAJOR;

  std::optional<std::reference_wrapper<::ttnn::Device>> outputDevice =
      std::nullopt;
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig = std::nullopt;

  if (not utils::inSystemMemory(op->out())) {
    // TODO (jnie): Update this once we support multi device tensors
    ::ttnn::Device &device =
        context.getDeviceFromView(op->device()->global_id(), 0);
    outputDevice = std::make_optional(std::ref(device));
    outputMemoryConfig =
        std::make_optional(utils::createMemoryConfig(op->out()));
  }

  ::ttnn::Tensor out =
      ::ttnn::full(shape, fillValue, outputDataType, outputLayout, outputDevice,
                   outputMemoryConfig);

  // When tensor is created with full op it is usually used as input to other
  // ops, not as output like in the case of empty op. That is why we really need
  // it to be in desired original layout, which we cannot deduce because of
  // issue #272, but it is more likely to be TILE layout so converting to that
  // one here. Unfortunately ttnn::to_layout works only with bfp16, so for other
  // types we are doing this hack which is safe to do only for single value
  // tensors.
  if (outputDataType == ::ttnn::DataType::BFLOAT16) {
    out =
        ::ttnn::to_layout(out, ::ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                          static_cast<::ttnn::Device *>(nullptr));
  } else if (shape.volume() == 1) {
    out.set_layout(::ttnn::Layout::TILE);
  }

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
