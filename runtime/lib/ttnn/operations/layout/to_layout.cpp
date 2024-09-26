// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_layout.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {

void run(const ::tt::target::ttnn::ToLayoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert((utils::isOnHost(inputTensor) or utils::isOnDevice(inputTensor)) &&
         "Unsupported storage type");
  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  assert(::tt::runtime::ttnn::utils::isValidTileShape(targetTileShape) &&
         "Invalid tile shape");
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

  // If device is specified, to_layout will send the host tensor to device
  // implicitly which is not desired
  ::ttnn::Tensor out =
      ::ttnn::to_layout(inputTensor, layout, std::nullopt, std::nullopt,
                        static_cast<::ttnn::Device *>(nullptr));

  const std::unordered_set<uint32_t> &programOutputs =
      tensorPool.getProgramOutputs();
  if (programOutputs.contains(op->out()->global_id())) {
    ::ttnn::Tensor &outputTensor = tensorPool.at(op->out()->global_id());
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(out);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = out.volume() * out.element_size();
    std::memcpy(dst, src, size);
  } else {
    tensorPool.insert_or_assign(op->out()->global_id(), out);
  }
}

} // namespace tt::runtime::ttnn::operations::layout
