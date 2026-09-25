// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/softmax_backward.h"
#include "metal/ops/softmax_backward/softmax_backward.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::SoftmaxBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &softmaxOutput =
      tensorPool.getTTNNTensorAndValidate(op->softmax_output());
  const ::ttnn::Tensor &grad = tensorPool.getTTNNTensorAndValidate(op->grad());

  ::ttnn::Tensor result =
      ::ttml::metal::softmax_backward(softmaxOutput, grad, op->dimension());
  tensorPool.insertTTNNTensorAndValidate(op->out(), result);
}

} // namespace tt::runtime::ttnn::operations::ttml
