// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/constant.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
template <typename T>
::tt::tt_metal::OwnedBuffer buffer(const ::flatbuffers::Vector<uint8_t> *data) {
  auto size = data->size() / sizeof(T);
  ::tt::tt_metal::owned_buffer::Buffer<T> ownedBuffer =
      tt::tt_metal::owned_buffer::create<T>(size);

  for (size_t i = 0; i < size; ++i) {
    ownedBuffer[i] = *data->GetAs<T>(i);
  }
  return ownedBuffer;
}

void run(const ::tt::target::ttnn::ConstantOp *op, ProgramContext &context) {
  ::ttnn::SimpleShape shape(::tt::runtime::ttnn::utils::toShapeFromFBShape(
      *op->out()->desc()->shape()));

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());

  const ::flatbuffers::Vector<uint8_t> *rawData = op->data();
  ::tt::tt_metal::OwnedBuffer owned_buffer;
  switch (dtype) {
  case ::ttnn::DataType::FLOAT32:
    owned_buffer = buffer<float>(rawData);
    break;
  case ::ttnn::DataType::UINT8:
    owned_buffer = buffer<uint8_t>(rawData);
    break;
  case ::ttnn::DataType::UINT16:
    owned_buffer = buffer<uint16_t>(rawData);
    break;
  case ::ttnn::DataType::INT32:
    owned_buffer = buffer<int32_t>(rawData);
    break;
  case ::ttnn::DataType::UINT32:
    owned_buffer = buffer<uint32_t>(rawData);
    break;
  case ::ttnn::DataType::BFLOAT16:
    owned_buffer = buffer<bfloat16>(rawData);
    break;
  default:
    LOG_FATAL("Unsupported data type");
  }

  ::ttnn::Tensor out(::tt::tt_metal::OwnedStorage{owned_buffer}, shape, dtype,
                     ::ttnn::Layout::ROW_MAJOR);

  context.getTensorPool().insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
