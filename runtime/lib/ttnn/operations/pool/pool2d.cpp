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
#include <ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp>

namespace tt::runtime::ttnn::operations::pool {

void runAvgPool2dOp(
    const ::tt::target::ttnn::Pool2dOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>, bool,
        bool, std::optional<int32_t>,
        const std::optional<const ::ttnn::MemoryConfig> &,
        const std::optional<const ::ttnn::TensorMemoryLayout>, bool)> &ttnnOp) {
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op->padding()->Get(0)),
                                static_cast<uint32_t>(op->padding()->Get(1))};
  } else {
    padding = std::array<uint32_t, 4>{
        static_cast<uint32_t>(op->padding()->Get(0)),  // top
        static_cast<uint32_t>(op->padding()->Get(2)),  // bottom
        static_cast<uint32_t>(op->padding()->Get(1)),  // left
        static_cast<uint32_t>(op->padding()->Get(3))}; // right
  }

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  ::ttnn::Tensor out =
      ttnnOp(input, op->batch_size(), op->input_height(), op->input_width(),
             op->channels(), kernelSize, stride, padding, op->ceil_mode(),
             op->extra_params_as_AvgPool2dExtraParams()->count_include_pad(),
             /*divisor_override=*/std::nullopt, outputMemoryConfig,
             appliedShardScheme, op->in_place_halo());

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void runMaxPool2dOp(
    const ::tt::target::ttnn::Pool2dOp *op, ProgramTensorPool &tensorPool,
    const std::function<std::variant<
        ::ttnn::Tensor, ::ttnn::operations::pool::MaxPoolWithIndicesResult>(
        const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>,
        std::array<uint32_t, 2>, bool,
        const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::TensorMemoryLayout> &, bool, bool)>
        &ttnnOp) {
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op->padding()->Get(0)),
                                static_cast<uint32_t>(op->padding()->Get(1))};
  } else {
    padding = std::array<uint32_t, 4>{
        static_cast<uint32_t>(op->padding()->Get(0)),  // top
        static_cast<uint32_t>(op->padding()->Get(2)),  // bottom
        static_cast<uint32_t>(op->padding()->Get(1)),  // left
        static_cast<uint32_t>(op->padding()->Get(3))}; // right
  }

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  auto result =
      ttnnOp(input, op->batch_size(), op->input_height(), op->input_width(),
             op->channels(), kernelSize, stride, padding, dilation,
             op->ceil_mode(), outputMemoryConfig, appliedShardScheme,
             op->in_place_halo(), false /* return_indices */);

  // Extract tensor from variant (we only care about the tensor, not indices for
  // runtime)
  ::ttnn::Tensor out = std::visit(
      [](const auto &val) -> ::ttnn::Tensor {
        if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                     ::ttnn::Tensor>) {
          return val;
        } else {
          return val.output; // MaxPoolWithIndicesResult case
        }
      },
      result);

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

void run(const ::tt::target::ttnn::GlobalAvgPool2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  // Call ttnn::global_avg_pool2d with input, memory_config, and output_dtype
  ::ttnn::Tensor out = ::ttnn::global_avg_pool2d(input, outputMemoryConfig,
                                                 /*output_dtype=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::pool
