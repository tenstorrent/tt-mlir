// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/scatter.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->lhs());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->rhs());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  // Use mask-based scatter workaround instead of using ttnn::scatter
  ::ttnn::Tensor onesLikeLhs = ::ttnn::ones_like(
      lhs, lhs.dtype(), lhs.layout(), std::nullopt, outputMemoryConfig);
  ::tt::tt_metal::Array4D startIndex = {0, 0, 0, 0};

  ::ttnn::Tensor indexPad =
      ::ttnn::pad(onesLikeLhs, rhs.padded_shape().to_array_4D(), startIndex, 0,
                  false, std::nullopt);

  ::ttnn::Tensor tempA = ::ttnn::pad(lhs, rhs.padded_shape().to_array_4D(),
                                     startIndex, 0, false, std::nullopt);

  ::ttnn::Tensor out = ::ttnn::where(indexPad, tempA, rhs);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
