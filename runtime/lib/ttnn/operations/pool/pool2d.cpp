// SPDX-FileCopyrightText: (c) 202 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/pool2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#include "ttnn/types.hpp"
#include <optional>
#include <ttnn/operations/functions.hpp>
#include <ttnn/operations/pool/generic/generic_pools.hpp>

namespace tt::runtime::ttnn::operations::pool {

void runAvgPool2dOp(
    const ::tt::target::ttnn::Pool2dOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>,
        std::array<uint32_t, 2>, bool, bool, std::optional<int32_t>,
        const std::optional<const ::ttnn::MemoryConfig> &,
        const std::optional<const ::ttnn::TensorMemoryLayout>, bool)> &ttnnOp) {
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride, padding;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->padding()->begin(), 2, padding.begin());

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  ::ttnn::Tensor out =
      ttnnOp(input, op->batch_size(), op->input_height(), op->input_width(),
             op->channels(), kernelSize, stride, padding, op->ceil_mode(),
             /*count_include_pad=*/true, /*divisor_override=*/std::nullopt,
             outputMemoryConfig, appliedShardScheme, op->in_place_halo());

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void runMaxPool2dOp(
    const ::tt::target::ttnn::Pool2dOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>, bool,
        const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::TensorMemoryLayout> &, bool)> &ttnnOp) {
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride, padding, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->padding()->begin(), 2, padding.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  ::ttnn::Tensor out = ttnnOp(
      input, op->batch_size(), op->input_height(), op->input_width(),
      op->channels(), kernelSize, stride, padding, dilation, op->ceil_mode(),
      outputMemoryConfig, appliedShardScheme, op->in_place_halo());

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::Pool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::Pool2dOpType::AvgPool2d: {
    runAvgPool2dOp(op, tensorPool, ::ttnn::avg_pool2d);
    break;
  }
  case ::tt::target::ttnn::Pool2dOpType::MaxPool2d: {
    runMaxPool2dOp(op, tensorPool, ::ttnn::max_pool2d);
  }
  }
}
} // namespace tt::runtime::ttnn::operations::pool
