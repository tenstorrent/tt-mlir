// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
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

  ::ttnn::Layout outputLayout [[maybe_unused]] =
      utils::inferLayoutFromTileShape(op->out());

  // TODO(bug #272), determine correct layout by tile shape in the future
  // currently tile shape is not set correctly, so as a workaround, hardcode
  // layout
  if (workaround::Env::get().ignoreTileShape) {
    outputLayout = ::ttnn::Layout::TILE;
  }

  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  if (workaround::Env::get().fullOpForceRowMajor) {
    outputLayout = ::ttnn::Layout::ROW_MAJOR;
  }

  std::optional<std::reference_wrapper<::ttnn::Device>> outputDevice =
      std::nullopt;
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig = std::nullopt;

  if (not utils::inSystemMemory(op->out())) {
    // TODO (jnie): Update this once we support multi device tensors
    ::ttnn::Device &device =
        context.getDeviceFromSubMesh(op->device()->global_id(), 0);
    outputDevice = std::make_optional(std::ref(device));
    outputMemoryConfig =
        std::make_optional(utils::createMemoryConfig(op->out()));
  }
  ::ttnn::Tensor out =
      ::ttnn::full(shape, fillValue, outputDataType, outputLayout, outputDevice,
                   outputMemoryConfig);
  if (tensorPool.isUserOutput(op->out()->global_id())) {
    tensorPool.copyTensorToUserOutput(out, op->out()->global_id());
  } else {
    tensorPool.insert_or_assign(op->out()->global_id(), out);
  }
}
} // namespace tt::runtime::ttnn::operations::creation
