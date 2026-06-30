// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv1d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

#include <array>
#include <variant>

namespace tt::runtime::ttnn::operations::conv {
using ::ttnn::Conv1dResult;
void run(const ::tt::target::ttnn::Conv1dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  LOG_ASSERT(op->padding()->size() == 2,
             "Padding expected to have 2 elements ([pL, pR])");

  // conv1d padding is a (left, right) pair, mapped to the std::array<2>
  // alternative of the TTNN variant.
  std::variant<std::array<uint32_t, 2>, uint32_t> padding;
  std::array<uint32_t, 2> lrPadding;
  std::copy_n(op->padding()->begin(), 2, lrPadding.begin());
  padding = lrPadding;

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::Conv1dConfig conv1dConfig;
  if (op->conv2d_config()) {
    conv1dConfig = utils::createConv2dConfig(op->conv2d_config());
  }

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

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

  std::optional<::ttnn::Conv1dSliceConfig> sliceConfig;
  if (op->conv2d_slice_config()) {
    sliceConfig = utils::createConv2dSliceConfig(op->conv2d_slice_config());
  }

  Conv1dResult result = ::ttnn::conv1d(
      input, weight, &targetDevice, op->in_channels(), op->out_channels(),
      op->batch_size(), op->input_length(), op->kernel_size(), op->stride(),
      padding, op->dilation(), op->groups(), outputDtype, bias, conv1dConfig,
      computeConfig, outputMemoryConfig, sliceConfig,
      /*return_output_dim=*/false, /*return_weights_and_bias=*/false);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
