// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/cross_entropy_fw.h"
#include "metal/ops/cross_entropy_fw/cross_entropy_fw.hpp" // ttml::metal::cross_entropy_fw

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::CrossEntropyForwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &target =
      tensorPool.getTTNNTensorAndValidate(op->target());

  ::ttnn::Tensor output = ::ttml::metal::cross_entropy_fw(input, target);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::ttml
