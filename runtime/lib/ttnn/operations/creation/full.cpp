// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/full.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
void runFullOp(const ::tt::target::ttnn::FullOp *op, auto &&fillValue,
               ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Shape shape = ::tt::runtime::ttnn::operations::utils::toTTNNShape(
      *op->out()->desc()->shape());
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());

  OptionalMeshDeviceRef meshDevice = std::nullopt;

  if (op->device()) {
    meshDevice = std::ref(context.getMeshDevice());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));

  ::ttnn::Tensor out =
      ::ttnn::full(shape, fillValue, dtype, layout, meshDevice, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::FullOp *op, ProgramContext &context) {
  if (op->fill_value_type() == ::tt::target::ttnn::FillValueType::FP) {
    runFullOp(op, op->fill_value_as_FP()->value(), context);
  } else if (op->fill_value_type() == ::tt::target::ttnn::FillValueType::I32) {
    runFullOp(op, op->fill_value_as_I32()->value(), context);
  } else {
    LOG_FATAL("unknown fill value type");
  }
}
} // namespace tt::runtime::ttnn::operations::creation
