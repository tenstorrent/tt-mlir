// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/concat.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ConcatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    const ::ttnn::Tensor &in = tensorPool.at(input->global_id());
    DEBUG_ASSERT(in.is_allocated());
    inputs.push_back(in);
  }
  int32_t dim = op->dim();
  ::ttnn::Tensor out = ::ttnn::concat(inputs, dim);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
