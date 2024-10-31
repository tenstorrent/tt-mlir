// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/workarounds.h"

namespace tt::runtime::ttnn::operations::binary {

void getEltwiseBinaryOPInputTensors(const ::tt::target::ttnn::EltwiseOp *op,
                                    ProgramTensorPool &tensorPool,
                                    ::ttnn::Tensor **lhs,
                                    ::ttnn::Tensor **rhs) {
  LOG_ASSERT(op->ins()->size() == 2, "Expected 2 inputs");
  *lhs = &(tensorPool.at(op->ins()->Get(0)->global_id()));
  *rhs = &(tensorPool.at(op->ins()->Get(1)->global_id()));
  DEBUG_ASSERT((*lhs)->is_allocated());
  DEBUG_ASSERT((*rhs)->is_allocated());

  // Switch the order of operands if the second operand requires broadcast
  // TODO(bug #1124): We're currently swapping the operands for binary ops
  // in runtime if the lhs operand is smaller (and requires broadcast onto the
  // rhs operand). We should add this check in the compiler.
  if (workaround::Env::get().swapBinaryOperands &&
      (*lhs)->volume() < (*rhs)->volume()) {
    std::swap(*lhs, *rhs);
  }
}

} // namespace tt::runtime::ttnn::operations::binary
