// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_layout.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToLayoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  // TODO (jnie): Update this once we support multi device tensors
  ::ttnn::Device &device =
      context.getDeviceFromView(op->device()->global_id(), 0);
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(inputTensor.is_allocated());
  assert((utils::isOnHost(inputTensor) or utils::isOnDevice(inputTensor)) &&
         "Unsupported storage type");

  ::ttnn::Layout layout;
  switch (op->layout()) {
  case ::tt::target::TensorLayout::RowMajor:
    layout = ::ttnn::Layout::ROW_MAJOR;
    break;
  case ::tt::target::TensorLayout::Tile:
    layout = ::ttnn::Layout::TILE;
    break;
  case ::tt::target::TensorLayout::Invalid:
    layout = ::ttnn::Layout::INVALID;
    break;
  }

  ::ttnn::Tensor out = ::ttnn::to_layout(inputTensor, layout, std::nullopt,
                                         std::nullopt, &device);

  tensorPool.try_emplace(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
