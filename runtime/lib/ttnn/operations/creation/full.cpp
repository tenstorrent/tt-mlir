// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/full.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::FullOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Shape shape = ::tt::runtime::ttnn::operations::utils::toTTNNShape(
      *op->out()->desc()->shape());
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
  float fillValue = op->fill_value();

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
} // namespace tt::runtime::ttnn::operations::creation
