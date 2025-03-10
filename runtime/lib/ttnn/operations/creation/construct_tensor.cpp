// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/construct_tensor.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::ConstructTensorOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Shape shape = ::tt::runtime::ttnn::operations::utils::toTTNNShape(
      *op->out()->desc()->shape());

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());

  // Calculate total size needed
  const uint32_t size = shape.volume();

  // Create buffer based on dtype
  ::tt::tt_metal::OwnedBuffer ownedBuffer;
  switch (dtype) {
  case ::ttnn::DataType::FLOAT32:
    ownedBuffer = tt::tt_metal::owned_buffer::create<float>(size);
    break;
  case ::ttnn::DataType::UINT8:
    ownedBuffer = tt::tt_metal::owned_buffer::create<uint8_t>(size);
    break;
  case ::ttnn::DataType::UINT16:
    ownedBuffer = tt::tt_metal::owned_buffer::create<uint16_t>(size);
    break;
  case ::ttnn::DataType::INT32:
    ownedBuffer = tt::tt_metal::owned_buffer::create<int32_t>(size);
    break;
  case ::ttnn::DataType::UINT32:
    ownedBuffer = tt::tt_metal::owned_buffer::create<uint32_t>(size);
    break;
  case ::ttnn::DataType::BFLOAT16:
    ownedBuffer = tt::tt_metal::owned_buffer::create<bfloat16>(size);
    break;
  default:
    LOG_FATAL("Unsupported data type");
  }

  ::ttnn::Tensor out(::tt::tt_metal::OwnedStorage{ownedBuffer}, shape, dtype,
                     ::ttnn::Layout::ROW_MAJOR);

  tensorPool.insertAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
