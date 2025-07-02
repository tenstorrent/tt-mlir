// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_layout.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToLayoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  LOG_ASSERT(::tt::runtime::ttnn::utils::isValidTileShape(targetTileShape),
             "Invalid tile shape");

  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  std::optional<::ttnn::DataType> dtype = std::nullopt;

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  ::ttnn::Tensor out =
      ::ttnn::to_layout(inputTensor, layout, dtype, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
