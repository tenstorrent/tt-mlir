// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/deletion/deallocate.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocateOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor &tensor = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(tensor.is_allocated());
  ::ttnn::deallocate(tensor, op->force());
  tensorPool.erase(op->in()->global_id());
}
} // namespace tt::runtime::ttnn::operations::deletion
