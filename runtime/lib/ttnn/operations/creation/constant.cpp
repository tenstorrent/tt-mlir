// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/constant.h"

#include "tt/runtime/detail/logger.h"

#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::ConstantOp *op, ProgramContext &context) {
  ::ttnn::Shape shape = utils::toTTNNShape(*op->out()->desc()->shape());

  // Get data type from tensor descriptor
  ::tt::target::DataType target_dtype =
      op->out()->desc()->layout()->memory_desc()->data_type();

  // Create buffer using shared utility
  ::tt::tt_metal::OwnedBuffer ownedBuffer =
      utils::createTypedBuffer(target_dtype, shape.volume(), op->data());

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());

  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()),
             "Output tensor is expected to be in system memory");
  LOG_ASSERT(!::tt::runtime::ttnn::operations::utils::isTilized(op->out()),
             "Output tensor is expected to be in ROW_MAJOR layout");

  ::ttnn::Tensor out(::tt::tt_metal::OwnedStorage(ownedBuffer), shape, dtype,
                     ::ttnn::Layout::ROW_MAJOR);

  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::creation
