// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/host_empty.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::HostEmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  std::vector<uint32_t> shape = ::tt::runtime::ttnn::utils::toShapeFromFBShape(
      *op->out()->desc()->shape());
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());

  // Calculate total size needed
  uint32_t size = 1;
  for (uint32_t dim : shape) {
    size *= dim;
  }

  // Create buffer based on dtype
  ::tt::tt_metal::OwnedBuffer owned_buffer;
  switch (dtype) {
  case ::ttnn::DataType::FLOAT32:
    owned_buffer = tt::tt_metal::owned_buffer::create<float>(size);
    break;
  case ::ttnn::DataType::UINT8:
    owned_buffer = tt::tt_metal::owned_buffer::create<uint8_t>(size);
    break;
  case ::ttnn::DataType::UINT16:
    owned_buffer = tt::tt_metal::owned_buffer::create<uint16_t>(size);
    break;
  case ::ttnn::DataType::INT32:
    owned_buffer = tt::tt_metal::owned_buffer::create<int32_t>(size);
    break;
  case ::ttnn::DataType::UINT32:
    owned_buffer = tt::tt_metal::owned_buffer::create<uint32_t>(size);
    break;
  case ::ttnn::DataType::BFLOAT16:
    owned_buffer = tt::tt_metal::owned_buffer::create<bfloat16>(size);
    break;
  default:
    LOG_FATAL("Unsupported data type");
  }

  ::ttnn::Tensor out(::tt::tt_metal::OwnedStorage{owned_buffer},
                     ::ttnn::SimpleShape(shape), dtype,
                     ::ttnn::Layout::ROW_MAJOR);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
