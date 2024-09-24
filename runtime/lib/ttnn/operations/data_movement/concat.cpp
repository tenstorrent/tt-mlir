// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ConcatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    inputs.push_back(tensorPool.at(input->global_id()));
  }
  int32_t dim = op->dim();
  ::ttnn::Tensor out = ::ttnn::concat(inputs, dim);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
