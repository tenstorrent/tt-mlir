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
  ::ttnn::DataType targetDataTypeTTNN = utils::getDataType(op->out());
  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  auto desiredLayout = ::ttnn::Layout::ROW_MAJOR;
  auto shape = ::ttnn::Shape(
      ::tt::tt_metal::Shape(::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->out()->desc()->shape())));

  ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
  ::ttnn::Tensor out =
      ::ttnn::empty(shape, targetDataTypeTTNN, desiredLayout, device);
  // use try emplace here so the program output tensor doesn't get overwritten
  tensorPool.try_emplace(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
