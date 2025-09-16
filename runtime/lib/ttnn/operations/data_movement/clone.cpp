// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/clone.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/data_movement/clone/clone.hpp"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::CloneOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  // Get and validate input tensor
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  // Extract optional dtype conversion
  std::optional<::ttnn::DataType> outputDtype = std::nullopt;
  if (op->dtype().has_value()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype().value());
  }

  // Create memory configuration for output tensor
  // Memory config is optional - only pass it if explicitly provided
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt;
  if (op->memory_config() != nullptr) {
    outputMemoryConfig = ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
        op->memory_config());
  }

  // Extract optional compute configuration
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config() != nullptr) {
    computeConfig =
        ::tt::runtime::ttnn::operations::utils::createDeviceComputeKernelConfig(
            op->compute_config());
  }

  // Execute clone operation
  ::ttnn::Tensor output =
      ::ttnn::clone(input, outputDtype, outputMemoryConfig, computeConfig);

  // Store output tensor in pool
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::data_movement
