// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.h"
#include "tt/runtime/detail/logger.h"

namespace tt::runtime::ttnn::operations::unary {

void getEltwiseUnaryOPInputTensor(const ::tt::target::ttnn::EltwiseOp *op,
                                  ProgramTensorPool &tensorPool,
                                  ::ttnn::Tensor **in) {
  LOG_ASSERT(op->ins()->size() == 1, "Expected 1 input, got ",
             op->ins()->size());
  *in = &(tensorPool.at(op->ins()->Get(0)->global_id()));
  DEBUG_ASSERT((*in)->is_allocated());
}

} // namespace tt::runtime::ttnn::operations::unary
