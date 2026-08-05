// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/cross_entropy_bw.h"
#include "metal/ops/cross_entropy_bw/cross_entropy_bw.hpp" // ttml::metal::cross_entropy_bw

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::CrossEntropyBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &target =
      tensorPool.getTTNNTensorAndValidate(op->target());
  const ::ttnn::Tensor &grad = tensorPool.getTTNNTensorAndValidate(op->grad());

  ::ttnn::Tensor output =
      ::ttml::metal::cross_entropy_bw(input, target, grad, op->scaler());

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::ttml
