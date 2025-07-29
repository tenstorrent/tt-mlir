// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/scatter.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::ScatterOp *op, ProgramContext &context) {
  LOG_ASSERT(::tt::runtime::workaround::Env::get().maskBasedScatter,
             "Mask-based scatter workaround is not enabled");

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &update =
      tensorPool.getTTNNTensorAndValidate(op->update());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  // Use mask-based scatter workaround instead of using ttnn::scatter
  ::ttnn::Tensor onesLikeLhs =
      ::ttnn::ones_like(update, update.dtype(), update.layout(), std::nullopt,
                        outputMemoryConfig);
  ::tt::tt_metal::Array4D startIndex = {0, 0, 0, 0};

  ::ttnn::Tensor indexPad = ::ttnn::pad(::ttnn::DefaultQueueId, onesLikeLhs,
                                        input.padded_shape().to_array_4D(),
                                        startIndex, 0, false, std::nullopt);

  ::ttnn::Tensor tempA = ::ttnn::pad(::ttnn::DefaultQueueId, update,
                                     input.padded_shape().to_array_4D(),
                                     startIndex, 0, false, std::nullopt);

  ::ttnn::Tensor out = ::ttnn::where(indexPad, tempA, input);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
