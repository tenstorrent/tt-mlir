// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

#include "ttnn/operations/experimental/quasar/conv2d/conv2d.hpp"

namespace tt::runtime::ttnn::operations::conv {
using ::ttnn::Conv2dResultWithOptions;
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  LOG_ASSERT(op->kernel_size()->size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(op->stride()->size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(op->padding()->size() == 2 || op->padding()->size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(op->dilation()->size() == 2,
             "Dilation expected to have 2 elements");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(op->padding()->begin(), 2, symPadding.begin());
    padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(op->padding()->begin(), 4, asymPadding.begin());
    padding = asymPadding;
  }

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::Conv2dConfig conv2dConfig;
  if (op->conv2d_config()) {
    conv2dConfig = utils::createConv2dConfig(op->conv2d_config());
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

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
  if (op->conv2d_slice_config()) {
    sliceConfig = utils::createConv2dSliceConfig(op->conv2d_slice_config());
  }

  // Quasar reimplements conv2d; the mainline op's program spec rejects the
  // Quasar compute config (TT_FATAL on holds_alternative<ComputeGen2Config> in
  // tt_metal/impl/metal2_host_api/program_spec.cpp). The Quasar entry point
  // takes the same arguments -- Conv2dConfig and Conv2dSliceConfig are the same
  // underlying types -- but returns its own, structurally identical, result
  // variant, so each branch unwraps its own.
  ::ttnn::Tensor out;
  if (utils::isQuasar()) {
    // NOTE: this does not work yet. Quasar conv2d currently fails inside
    // tt-metal with "Trying to construct a Gen2 compute config but the
    // kernel's ComputeHardwareConfig does not hold a ComputeGen2Config"
    // (tt_metal/impl/metal2_host_api/program_spec.cpp). Verified that passing
    // compute_config_ = std::nullopt does not avoid it, so the Gen1/Gen2
    // mismatch is inside the Quasar conv path rather than in what we pass.
    // Left wired up so the next attempt starts from the real error.
    auto result = ::ttnn::operations::experimental::quasar::conv2d(
        input, weight, &targetDevice, op->in_channels(), op->out_channels(),
        op->batch_size(), op->input_height(), op->input_width(), kernelSize,
        stride, padding, dilation, op->groups(), outputDtype, bias,
        conv2dConfig, computeConfig, outputMemoryConfig, sliceConfig);
    LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));
    out = std::get<::ttnn::Tensor>(result);
  } else {
    Conv2dResultWithOptions result = ::ttnn::conv2d(
        input, weight, &targetDevice, op->in_channels(), op->out_channels(),
        op->batch_size(), op->input_height(), op->input_width(), kernelSize,
        stride, padding, dilation, op->groups(), outputDtype, bias,
        conv2dConfig, computeConfig, outputMemoryConfig, sliceConfig);
    LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));
    out = std::get<::ttnn::Tensor>(result);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
