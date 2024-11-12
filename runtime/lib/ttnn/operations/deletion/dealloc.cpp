// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "dealloc.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "ttnn/operations/core/core.hpp"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor &tensor = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(tensor.is_allocated());
  ::ttnn::operations::core::deallocate(tensor);

  // The tensor should be deallocated after the deallocate call.
  // Still this assert may be hit in the future for multidevice/async ttnn
  // support. In that case, we will reevaluate the assert/dealloc behaviour and
  // adjust it accordingly.
  //
  DEBUG_ASSERT(!tensor.is_allocated());
  tensorPool.erase(op->in()->global_id());
}
} // namespace tt::runtime::ttnn::operations::deletion
