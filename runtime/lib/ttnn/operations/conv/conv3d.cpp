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

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv3dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  LOG_ASSERT(op->kernel_size()->size() == 3,
             "Kernel size expected to have 3 elements");
  LOG_ASSERT(op->stride()->size() == 3, "Stride expected to have 3 elements");
  LOG_ASSERT(op->padding()->size() == 3, "Padding expected to have 3 elements");

  std::array<uint32_t, 3> kernelSize, stride, padding;
  std::copy_n(op->kernel_size()->begin(), 3, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 3, stride.begin());
  std::copy_n(op->padding()->begin(), 3, padding.begin());

  ::ttnn::DataType outputDtype = ::ttnn::DataType::BFLOAT16;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  // Conv3dConfig is set at compile-time by Conv3dBlockingRewritePattern.
  // Just read from flatbuffer and construct the config.
  ::ttnn::operations::experimental::conv3d::Conv3dConfig conv3dConfig;
  if (op->conv3d_config()) {
    const auto *fbConfig = op->conv3d_config();
    if (fbConfig->weights_dtype()) {
      conv3dConfig.weights_dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
          *fbConfig->weights_dtype());
    }
    if (fbConfig->t_out_block()) {
      conv3dConfig.T_out_block = *fbConfig->t_out_block();
    }
    if (fbConfig->w_out_block()) {
      conv3dConfig.W_out_block = *fbConfig->w_out_block();
    }
    if (fbConfig->h_out_block()) {
      conv3dConfig.H_out_block = *fbConfig->h_out_block();
    }
    if (fbConfig->c_out_block()) {
      conv3dConfig.C_out_block = *fbConfig->c_out_block();
    }
    if (fbConfig->c_in_block()) {
      conv3dConfig.C_in_block = *fbConfig->c_in_block();
    }
    if (const auto *gridCoord = fbConfig->compute_with_storage_grid_size()) {
      conv3dConfig.compute_with_storage_grid_size =
          tt::tt_metal::CoreCoord{gridCoord->x(), gridCoord->y()};
    } else {
      // Fallback: Use device grid size if not set at compile-time
      conv3dConfig.compute_with_storage_grid_size =
          targetDevice.compute_with_storage_grid_size();
    }
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  auto deviceComputeConfig = ::ttnn::init_device_compute_kernel_config(
      targetDevice.arch(), computeConfig, MathFidelity::HiFi4, true, true,
      false);

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::experimental::conv3d(
      input, weight, bias, conv3dConfig, outputDtype, op->out_channels(),
      kernelSize, stride, padding, std::array<uint32_t, 3>{1, 1, 1},
      op->padding_mode()->str(), op->groups(), outputMemoryConfig,
      deviceComputeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
