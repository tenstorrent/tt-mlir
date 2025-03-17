// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/ttnn/operations/eltwise/ternary/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::ternary {

void getEltwiseTernaryOpInputTensors(const ::tt::target::ttnn::EltwiseOp *op,
                                     ProgramTensorPool &tensorPool,
                                     ::ttnn::Tensor **first,
                                     ::ttnn::Tensor **second,
                                     ::ttnn::Tensor **third) {
  LOG_ASSERT(op->ins()->size() == 3, "Expected 3 inputs");
  *first = &(tensorPool.getAndValidate(op->ins()->Get(0)));
  *second = &(tensorPool.getAndValidate(op->ins()->Get(1)));
  *third = &(tensorPool.getAndValidate(op->ins()->Get(2)));
}

} // namespace tt::runtime::ttnn::operations::ternary
