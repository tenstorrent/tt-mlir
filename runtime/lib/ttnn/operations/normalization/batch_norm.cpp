// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/batch_norm.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::batch_norm {
void run(const ::tt::target::ttnn::BatchNormInferenceOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  float epsilon = op->epsilon();
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  // For inference: training=false, momentum=0.1 (default, not used in
  // inference)
  ::ttnn::Tensor output = ::ttnn::batch_norm(
      input, runningMean, runningVar, false, epsilon, 0.1f, weight, bias,
      /*output=*/std::nullopt, /*memory_config=*/std::nullopt, computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::BatchNormTrainingOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  float epsilon = op->epsilon();
  float momentum = op->momentum();

  // For training: training=true
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::Tensor output = ::ttnn::batch_norm(
      input, runningMean, runningVar, true, epsilon, momentum, weight, bias,
      /*output=*/std::nullopt, /*memory_config=*/std::nullopt, computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::batch_norm
