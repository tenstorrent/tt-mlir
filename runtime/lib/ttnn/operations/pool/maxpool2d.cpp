// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/maxpool2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/types.hpp"
#include <optional>

namespace tt::runtime::ttnn::operations::pool {

void run(const ::tt::target::ttnn::MaxPool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor input = tensorPool.getAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride, padding, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->padding()->begin(), 2, padding.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  ::ttnn::Tensor out = ::ttnn::max_pool2d(
      input, op->batch_size(), op->input_height(), op->input_width(),
      op->channels(), kernelSize, stride, padding, dilation, outputMemoryConfig,
      /*applied_shard_scheme=*/std::nullopt, op->ceil_mode());

  tensorPool.insertAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::pool
