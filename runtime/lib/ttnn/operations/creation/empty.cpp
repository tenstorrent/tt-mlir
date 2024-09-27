// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "empty.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::DataType dtype = utils::getDataType(op->out());
  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  ::ttnn::Layout layout __attribute__((unused)) =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());
  layout = ::ttnn::Layout::ROW_MAJOR;
  ::ttnn::Shape shape = ::ttnn::Shape(::tt::tt_metal::LegacyShape(
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->out()->desc()->shape())));
  ::ttnn::Tensor out;
  if (not utils::inSystemMemory(op->out())) {
    // TODO (jnie): Update this once we support multi device tensors
    ::ttnn::Device &device =
        context.getDeviceFromView(op->device()->global_id(), 0);
    ::ttnn::MemoryConfig memoryConfig =
        utils::createMemoryConfig(op->memcfg(), op->out());
    out = ::ttnn::empty(shape, dtype, layout, &device, memoryConfig);
  } else {
    out = ::ttnn::zeros(shape, dtype, layout);
  }

  // TODO(mrakita): Revert !!!
  out.set_layout(::ttnn::Layout::TILE);

  tensorPool.try_emplace(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
