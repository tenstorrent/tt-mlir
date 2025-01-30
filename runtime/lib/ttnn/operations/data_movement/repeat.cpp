// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/repeat.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::RepeatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());
  const auto *fbShape = op->repeat_dims();
  const std::vector<uint32_t> repeatDims(fbShape->begin(), fbShape->end());
  ::ttnn::Shape repeatDimsShape(repeatDims);
  ::ttnn::Tensor out = ::ttnn::repeat(in, repeatDimsShape);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
