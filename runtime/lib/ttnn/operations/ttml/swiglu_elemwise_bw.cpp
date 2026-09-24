// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/swiglu_elemwise_bw.h"
#include "metal/ops/swiglu_elemwise_bw/swiglu_elemwise_bw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::SwigluElemwiseBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &gate = tensorPool.getTTNNTensorAndValidate(op->gate());
  const ::ttnn::Tensor &gradOutput =
      tensorPool.getTTNNTensorAndValidate(op->grad_output());

  // The preallocated output operands are left unset: the tensor pool owns the
  // result buffers.
  ::ttml::metal::SwigluElemwiseBwResult result =
      ::ttml::metal::swiglu_elemwise_bw(input, gate, gradOutput);

  tensorPool.insertTTNNTensorAndValidate(op->grad_input(), result.dL_dlinear1);
  tensorPool.insertTTNNTensorAndValidate(op->grad_gate(), result.dL_dgate);
}

} // namespace tt::runtime::ttnn::operations::ttml
