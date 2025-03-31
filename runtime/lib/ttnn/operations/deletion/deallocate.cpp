// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/deletion/deallocate.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocateOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor &tensor = tensorPool.getTTNNTensorAndValidate(op->in());

  bool shouldDeallocate =
      tensor.storage_type() != ::ttnn::StorageType::BORROWED &&
      (op->force() ||
       tensorPool.getRuntimeTensorUseCount(op->in()->global_id()) == 1);

  if (shouldDeallocate) {
    ::ttnn::deallocate(tensor, op->force());
  }
  tensorPool.erase(op->in());
}
} // namespace tt::runtime::ttnn::operations::deletion
