// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/ttnn/operations/eltwise/unary/utils.h"
#include "tt/runtime/detail/logger.h"

namespace tt::runtime::ttnn::operations::unary {

void getEltwiseUnaryOpInputTensor(const ::tt::target::ttnn::EltwiseOp *op,
                                  ProgramTensorPool &tensorPool,
                                  ::ttnn::Tensor **in) {
  LOG_ASSERT(op->ins()->size() == 1, "Expected 1 input, got ",
             op->ins()->size());
  *in = &(tensorPool.getTTNNTensorAndValidate(op->ins()->Get(0)));
}

} // namespace tt::runtime::ttnn::operations::unary
