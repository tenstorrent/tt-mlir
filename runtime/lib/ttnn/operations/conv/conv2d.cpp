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

#include <unordered_map>

namespace tt::runtime::ttnn::operations::conv {

// Issue #7414: Cache for prepared conv2d weights/bias. On the first conv2d
// call, we ask conv2d to return the prepared weight/bias it computed (with the
// correct device allocator state). On subsequent calls we reuse them, avoiding
// both the PCC mismatch from consteval preparation and the perf cost of
// re-preparing every time.
struct CachedConv2dPrepare {
  ::ttnn::Tensor weight;
  std::optional<::ttnn::Tensor> bias;
};
static std::unordered_map<const ::tt::target::ttnn::Conv2dOp *,
                           CachedConv2dPrepare>
    conv2dPrepareCache;

using ::ttnn::Conv2dResultWithOptions;
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor weight = tensorPool.getTTNNTensorAndValidate(op->weight());

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

  // Issue #7414: Use cached prepared weight/bias from a previous conv2d call
  // that computed them with the correct device allocator state.
  auto cacheIt = conv2dPrepareCache.find(op);
  if (cacheIt != conv2dPrepareCache.end()) {
    weight = cacheIt->second.weight;
    if (cacheIt->second.bias) {
      bias = cacheIt->second.bias;
    }
  }

  // Ask conv2d to return prepared weight/bias so we can cache them.
  bool requestWeights = (cacheIt == conv2dPrepareCache.end());

  Conv2dResultWithOptions result = ::ttnn::conv2d(
      input, weight, &targetDevice, op->in_channels(), op->out_channels(),
      op->batch_size(), op->input_height(), op->input_width(), kernelSize,
      stride, padding, dilation, op->groups(), outputDtype, bias, conv2dConfig,
      computeConfig, outputMemoryConfig, sliceConfig,
      /*return_output_dim=*/false,
      /*return_weights_and_bias=*/requestWeights);

  // Extract output tensor and cache prepared weight/bias if returned.
  ::ttnn::Tensor out;
  using WeightBias = std::tuple<::ttnn::Tensor, std::optional<::ttnn::Tensor>>;
  using OutHW = std::tuple<uint32_t, uint32_t>;

  if (auto *v = std::get_if<::ttnn::Tensor>(&result)) {
    out = *v;
  } else if (auto *v =
                 std::get_if<std::tuple<::ttnn::Tensor, WeightBias>>(&result)) {
    out = std::get<0>(*v);
    auto &[w, b] = std::get<1>(*v);
    conv2dPrepareCache[op] = {w, b};
  } else if (auto *v =
                 std::get_if<std::tuple<::ttnn::Tensor, OutHW>>(&result)) {
    out = std::get<0>(*v);
  } else if (auto *v = std::get_if<
                 std::tuple<::ttnn::Tensor, OutHW, WeightBias>>(&result)) {
    out = std::get<0>(*v);
    auto &[w, b] = std::get<2>(*v);
    conv2dPrepareCache[op] = {w, b};
  } else {
    LOG_ASSERT(false, "Unexpected Conv2dResultWithOptions variant");
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
