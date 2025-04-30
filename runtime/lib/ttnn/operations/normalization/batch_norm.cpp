// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/batch_norm.h"

namespace tt::runtime::ttnn::operations::batch_norm {
void run(const ::tt::target::ttnn::BatchNormOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  ::ttnn::Tensor &running_mean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &running_var =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  ::ttnn::Tensor &weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  bool training = op->training();
  float epsilon = op->epsilon();

  ::ttnn::Tensor output =
      ::ttnn::batch_norm(input, running_mean, running_var, training, epsilon,
                         /* momentum = */ 0.1, weight, bias);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::batch_norm
