// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv3d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

namespace {
::ttnn::operations::experimental::conv3d::Conv3dConfig
createConv3dConfig(const ::tt::target::ttnn::Conv3dOp *op,
                   const std::array<uint32_t, 3> &kernelSize,
                   const std::array<uint32_t, 3> &stride,
                   const std::array<uint32_t, 3> &padding,
                   const std::optional<::ttnn::DataType> &outputDtype,
                   ::ttnn::MeshDevice &targetDevice) {
  ::ttnn::operations::experimental::conv3d::Conv3dConfig config;

  config.dtype = outputDtype.value_or(::ttnn::DataType::BFLOAT16);
  config.weights_dtype = ::ttnn::DataType::BFLOAT16;
  config.output_layout = ::ttnn::Layout::ROW_MAJOR;

  config.output_channels = op->out_channels();
  config.kernel_size = kernelSize;
  config.stride = stride;
  config.padding = padding;
  config.padding_mode = op->padding_mode()->str();
  config.groups = op->groups();

  config.compute_with_storage_grid_size =
      targetDevice.compute_with_storage_grid_size();

  return config;
}
} // namespace

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv3dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias;
  if (const auto *bias_value = op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(bias_value);
  }

  LOG_ASSERT(op->kernel_size()->size() == 3,
             "Kernel size expected to have 3 elements");
  LOG_ASSERT(op->stride()->size() == 3, "Stride expected to have 3 elements");
  LOG_ASSERT(op->padding()->size() == 3, "Padding expected to have 3 elements");

  std::array<uint32_t, 3> kernelSize, stride, padding;
  std::copy_n(op->kernel_size()->begin(), 3, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 3, stride.begin());
  std::copy_n(op->padding()->begin(), 3, padding.begin());

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  auto config = createConv3dConfig(op, kernelSize, stride, padding, outputDtype,
                                   targetDevice);

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (const auto *compute_config_value = op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(compute_config_value);
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::experimental::conv3d(
      input, weight, bias, config, outputMemoryConfig, computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
