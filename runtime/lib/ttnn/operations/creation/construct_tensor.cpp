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
  ::tt::target::DataType targetDType =
      op->out()->desc()->layout()->memory_desc()->data_type();
  ::tt::tt_metal::OwnedBuffer ownedBuffer =
      utils::createTypedBuffer(targetDType, size);

  ::ttnn::Tensor out(::tt::tt_metal::OwnedStorage{ownedBuffer}, shape, dtype,
                     ::ttnn::Layout::ROW_MAJOR);

  tensorPool.insertAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
