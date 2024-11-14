// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/workarounds.h"

namespace tt::runtime::ttnn::operations::ternary {

void getEltwiseTernaryOPInputTensors(const ::tt::target::ttnn::EltwiseOp *op,
                                     ProgramTensorPool &tensorPool,
                                     ::ttnn::Tensor **first,
                                     ::ttnn::Tensor **second,
                                     ::ttnn::Tensor **third) {
  LOG_ASSERT(op->ins()->size() == 3, "Expected 3 inputs");
  *first = &(tensorPool.at(op->ins()->Get(0)->global_id()));
  *second = &(tensorPool.at(op->ins()->Get(1)->global_id()));
  *third = &(tensorPool.at(op->ins()->Get(2)->global_id()));
  DEBUG_ASSERT((*first)->is_allocated());
  DEBUG_ASSERT((*second)->is_allocated());
  DEBUG_ASSERT((*third)->is_allocated());
}

} // namespace tt::runtime::ttnn::operations::ternary
