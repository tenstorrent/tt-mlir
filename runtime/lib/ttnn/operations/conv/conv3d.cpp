// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv3d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv3dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  LOG_ASSERT(op->kernel_size()->size() == 3,
             "Kernel size expected to have 3 elements");
  LOG_ASSERT(op->stride()->size() == 3, "Stride expected to have 3 elements");
  LOG_ASSERT(op->padding()->size() == 3,
             "Padding expected to have 3 elements");

  std::array<uint32_t, 3> kernelSize, stride, padding;
  std::copy_n(op->kernel_size()->begin(), 3, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 3, stride.begin());
  std::copy_n(op->padding()->begin(), 3, padding.begin());

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::operations::experimental::conv3d::Conv3dConfig config;

  // Data types
  config.dtype = outputDtype.value_or(::ttnn::DataType::BFLOAT16);
  config.weights_dtype = ::ttnn::DataType::BFLOAT16;
  config.output_layout = ::ttnn::Layout::ROW_MAJOR;

  // Blocking parameters - use defaults that work for most shapes
  // 0 = auto-select optimal values based on tensor shapes
  config.T_out_block = 1;
  config.W_out_block = 1;
  config.H_out_block = 1;
  config.C_out_block = 0;
  config.C_in_block = 0;

  // Convolution parameters
  config.output_channels = op->out_channels();
  config.kernel_size = kernelSize;
  config.stride = stride;
  config.padding = padding;
  config.padding_mode = op->padding_mode()->str();
  config.groups = op->groups();

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  // Query device capabilities for grid size
  config.compute_with_storage_grid_size =
      targetDevice.compute_with_storage_grid_size();

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::operations::experimental::conv3d::invoke(
      input, weight, bias, config, outputMemoryConfig, computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
