// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "dealloc.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocOp *op, ProgramContext &context) {
  bool force = true;
  ProgramTensorPool &tensorPool = context.tensorPool;
  ::ttnn::Tensor &tensor = tensorPool.at(op->in()->global_id());
  tensor.deallocate(force);
  tensorPool.erase(op->in()->global_id());
}
} // namespace tt::runtime::ttnn::operations::deletion
