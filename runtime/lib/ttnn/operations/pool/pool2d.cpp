// SPDX-FileCopyrightText: (c) 202 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/pool2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#include "ttnn/types.hpp"
#include <optional>
#include <ttnn/operations/functions.hpp>
#include <ttnn/operations/pool/generic/generic_pools.hpp>

namespace tt::runtime::ttnn::operations::pool {

void runPool2dOp(const ::tt::target::ttnn::Pool2dOp *op,
                 ProgramTensorPool &tensorPool,
                 const std::function<::ttnn::Tensor(
                     const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t,
                     uint32_t, std::array<uint32_t, 2>, std::array<uint32_t, 2>,
                     std::array<uint32_t, 2>, std::array<uint32_t, 2>,
                     const std::optional<::ttnn::MemoryConfig> &,
                     const std::optional<::ttnn::TensorMemoryLayout> &, bool,
                     bool)> &ttnnOp) {
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());

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

  ::ttnn::Tensor out = ttnnOp(input, op->batch_size(), op->input_height(),
                              op->input_width(), op->channels(), kernelSize,
                              stride, padding, dilation, outputMemoryConfig,
                              /*applied_shard_scheme=*/std::nullopt,
                              op->ceil_mode(), /*in_place_halo=*/false);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
void run(const ::tt::target::ttnn::Pool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::Pool2dOpType::AvgPool2d: {
    runPool2dOp(op, tensorPool, ::ttnn::avg_pool2d);
    break;
  }
  case ::tt::target::ttnn::Pool2dOpType::MaxPool2d: {
    runPool2dOp(op, tensorPool, ::ttnn::max_pool2d);
  }
  }
}
} // namespace tt::runtime::ttnn::operations::pool
