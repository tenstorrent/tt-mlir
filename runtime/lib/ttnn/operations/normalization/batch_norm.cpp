// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/batch_norm.h"

namespace tt::runtime::ttnn::operations::batch_norm {
void run(const ::tt::target::ttnn::BatchNormInferenceOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  ::ttnn::Tensor &weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  float epsilon = op->epsilon();

  // For inference: training=false, momentum=0.1 (default, not used in
  // inference)
  ::ttnn::Tensor output = ::ttnn::batch_norm(
      input, runningMean, runningVar, false, epsilon, 0.1f, weight, bias);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::BatchNormTrainingOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  ::ttnn::Tensor &weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  float epsilon = op->epsilon();
  float momentum = op->momentum();

  // For training: training=true
  ::ttnn::Tensor output = ::ttnn::batch_norm(
      input, runningMean, runningVar, true, epsilon, momentum, weight, bias);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);

  // Note: ttnn::batch_norm with training=true updates running_mean and
  // running_var in-place.
  tensorPool.insertTTNNTensorAndValidate(op->batch_mean(), runningMean);
  tensorPool.insertTTNNTensorAndValidate(op->batch_variance(), runningVar);
}
} // namespace tt::runtime::ttnn::operations::batch_norm
