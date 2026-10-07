// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/silu_bw.h"
#include "metal/ops/silu_bw/silu_bw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::SiluBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &gradOutput =
      tensorPool.getTTNNTensorAndValidate(op->grad_output());

  // The preallocated output operand is left unset: the tensor pool owns the
  // result buffer.
  ::ttnn::Tensor gradInput = ::ttml::metal::silu_bw(input, gradOutput);

  tensorPool.insertTTNNTensorAndValidate(op->grad_input(), gradInput);
}

} // namespace tt::runtime::ttnn::operations::ttml
