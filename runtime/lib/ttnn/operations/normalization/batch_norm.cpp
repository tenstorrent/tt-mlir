// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/batch_norm.h"

namespace tt::runtime::ttnn::operations::batch_norm {
void run(const ::tt::target::ttnn::BatchNormOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  ::ttnn::Tensor &weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  bool training = op->training();
  float epsilon = op->epsilon();
  float momentum = op->momentum();

  ::ttnn::Tensor output =
      ::ttnn::batch_norm(input, runningMean, runningVar, training, epsilon,
                         momentum, weight, bias);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::batch_norm
