// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_layout.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {

// static void validate(const ::tt::target::TensorRef *inputTensor,
//                                           const ::tt::target::TensorRef
//                                           *outputTensor) {
//   assert(utils::getDataType(inputTensor) == utils::getDataType(outputTensor)
//   &&
//          "Input and output tensor data types must match, insert typecast op
//          to convert data types");

//   assert(utils::createMemoryConfig(inputTensor) ==
//   utils::createMemoryConfig(outputTensor) &&
//          "Input and output tensor memory configs must match, insert
//          to_memory_config op to convert memory configs");

//   assert(utils::getMemorySpace(inputTensor) ==
//   utils::getMemorySpace(outputTensor) &&
//          "Input and output tensor memory spaces must match, insert
//          to_device/from_device ops to convert memory spaces");
// }

void run(const ::tt::target::ttnn::ToLayoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::tt::target::TensorRef *inputTensorRef = op->in();
  const ::tt::target::TensorRef *outputTensorRef = op->out();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
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

  ::ttnn::Tensor out =
      ::ttnn::to_layout(inputTensor, layout, std::nullopt, std::nullopt,
                        static_cast<::ttnn::Device *>(nullptr));

  tensorPool.try_emplace(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
