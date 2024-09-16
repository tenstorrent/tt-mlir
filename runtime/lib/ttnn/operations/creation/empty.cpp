// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "empty.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  ::ttnn::DataType dtype = utils::toTTNNDataType(op->dtype());
  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  ::ttnn::Layout layout __attribute__((unused)) =
      utils::toTTNNLayout(op->layout());
  layout = ::ttnn::Layout::ROW_MAJOR;
  ::ttnn::Shape shape = ::ttnn::Shape(
      Shape(utils::toShapeFromFBShape(*op->out()->desc()->shape())));

  const tt::target::DeviceRef *device = op->device();
  ::ttnn::Tensor out;
  if (device) {
    ::ttnn::MemoryConfig memoryConfig =
        utils::createMemoryConfig(op->memcfg(), op->out());
    out = ::ttnn::empty(shape, dtype, layout, getDevice(device, devicePool),
                        memoryConfig);
  } else {
    out = ::ttnn::empty(shape, dtype, layout);
  }

  tensorPool.try_emplace(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
