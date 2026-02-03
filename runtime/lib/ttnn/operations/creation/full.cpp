// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/full.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
namespace {
void run(const ::tt::target::ttnn::FullOp *op, auto &&fillValue,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Shape shape =
      ::tt::runtime::ttnn::operations::utils::toTTNNShape(*op->shape());

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
  if (op->layout()) {
    layout = ::tt::runtime::ttnn::utils::toTTNNLayout(*(op->layout()));
  }

  OptionalMeshDeviceRef meshDevice;
  if (op->device()) {
    meshDevice = std::ref(context.getMeshDevice());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  if (op->memcfg()) {
    memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  }

  ::ttnn::Tensor out =
      ::ttnn::full(shape, fillValue, dtype, layout, meshDevice, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace

void run(const ::tt::target::ttnn::FullOp *op, ProgramContext &context) {
  switch (op->fill_value_type()) {
  case ::tt::target::ttnn::NumberType::FP:
    run(op, op->fill_value_as_FP()->value(), context);
    break;
  case ::tt::target::ttnn::NumberType::I32:
    run(op, op->fill_value_as_I32()->value(), context);
    break;
  default:
    LOG_FATAL("unknown fill value type");
  }
}
} // namespace tt::runtime::ttnn::operations::creation
