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

  std::optional<::ttnn::experimental::prim::Conv3dConfig> conv3dConfig;
  if (op->conv3d_config()) {
    const auto *fbConfig = op->conv3d_config();
    ::ttnn::experimental::prim::Conv3dConfig config;
    if (fbConfig->weights_dtype()) {
      config.weights_dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
          *fbConfig->weights_dtype());
    }
    if (fbConfig->t_out_block()) {
      config.T_out_block = *fbConfig->t_out_block();
    }
    if (fbConfig->w_out_block()) {
      config.W_out_block = *fbConfig->w_out_block();
    }
    if (fbConfig->h_out_block()) {
      config.H_out_block = *fbConfig->h_out_block();
    }
    if (fbConfig->c_out_block()) {
      config.C_out_block = *fbConfig->c_out_block();
    }
    if (fbConfig->c_in_block()) {
      config.C_in_block = *fbConfig->c_in_block();
    }
    if (const auto *gridCoord = fbConfig->compute_with_storage_grid_size()) {
      config.compute_with_storage_grid_size =
          tt::tt_metal::CoreCoord{gridCoord->x(), gridCoord->y()};
    } else {
      config.compute_with_storage_grid_size =
          targetDevice.compute_with_storage_grid_size();
    }
    conv3dConfig = config;
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  const uint32_t l1Alignment = ::tt::tt_metal::hal::get_l1_alignment();
  constexpr uint32_t kTileWidth = tt::constants::TILE_WIDTH;
  const uint32_t paddedOutChannels =
      tt::round_up(op->out_channels(), kTileWidth);
  const bool hasRequestedCInBlock =
      op->conv3d_config() && op->conv3d_config()->c_in_block();
  const bool hasRequestedCOutBlock =
      op->conv3d_config() && op->conv3d_config()->c_out_block();
  const uint32_t requestedCInBlock =
      hasRequestedCInBlock ? *op->conv3d_config()->c_in_block() : 0;
  const uint32_t requestedCOutBlock =
      hasRequestedCOutBlock ? *op->conv3d_config()->c_out_block() : 0;
  const uint32_t constrainedRequestedCOutBlock =
      requestedCOutBlock == 0
          ? 0
          : tt::round_up(requestedCOutBlock, kTileWidth);
  const bool cOutDividesPaddedOut =
      constrainedRequestedCOutBlock > 0 &&
      (paddedOutChannels % constrainedRequestedCOutBlock == 0);
  const uint32_t selectedCInBlock =
      requestedCInBlock > 0 ? requestedCInBlock : l1Alignment;
  const uint32_t selectedCOutBlock =
      cOutDividesPaddedOut
          ? constrainedRequestedCOutBlock
          : kTileWidth;
  if (conv3dConfig) {
    if (hasRequestedCInBlock) {
      conv3dConfig->C_in_block = selectedCInBlock;
    }
    if (hasRequestedCOutBlock) {
      conv3dConfig->C_out_block = selectedCOutBlock;
    }
  }

  auto deviceComputeConfig = ::ttnn::init_device_compute_kernel_config(
      targetDevice.arch(), computeConfig, ::tt::tt_metal::MathFidelity::HiFi4,
      true, true, false);

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::experimental::conv3d(
      input, weight, &targetDevice, bias, conv3dConfig, outputDtype,
      op->out_channels(), kernelSize, stride, padding,
      std::array<uint32_t, 3>{1, 1, 1}, op->padding_mode()->str(), op->groups(),
      outputMemoryConfig, deviceComputeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
