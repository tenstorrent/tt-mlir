// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/constant.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <ttnn/types.hpp>

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::ConstantOp *op, ProgramContext &context) {
  ::ttnn::Shape shape =
      operations::utils::toTTNNShape(*op->out()->desc()->shape());

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

  ::ttnn::MeshDevice *meshDevice = nullptr;
  if (op->device()) {
    meshDevice = context.getMeshDevicePtr().get();
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  if (op->memcfg()) {
    memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  }

  ::ttnn::Tensor out =
      utils::toTTNNTensor(op->data(), shape, dtype, meshDevice, layout,
                          memoryConfig.value_or(::ttnn::DRAM_MEMORY_CONFIG));

  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::creation
