// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ConcatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    const ::ttnn::Tensor &in = tensorPool.at(input->global_id());
    std::cout<<"concat input logical shape: "<<in.get_logical_shape()<<std::endl;
    DEBUG_ASSERT(in.is_allocated());
    inputs.push_back(in);
  }
  int32_t dim = op->dim();
  ::ttnn::Tensor out = ::ttnn::concat(inputs, dim);
  std::cout<<"concat output logical shape: "<<out.get_logical_shape()<<std::endl;
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
