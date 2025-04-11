// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/deletion/deallocate.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocateOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensorPool.getTTNNTensorWrapperAndValidate(op->in());
  ::ttnn::Tensor &ttnnTensor = tensorWrapper.getTensor();

  bool shouldDeallocate =
      (ttnnTensor.storage_type() != ::ttnn::StorageType::BORROWED) &&
      (!tensorWrapper.shouldRetain());

  if (shouldDeallocate) {
    ::ttnn::deallocate(ttnnTensor, op->force());
  }

  tensorPool.erase(op->in());
}
} // namespace tt::runtime::ttnn::operations::deletion
