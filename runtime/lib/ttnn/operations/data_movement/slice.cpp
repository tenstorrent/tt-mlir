// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SliceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());
  ::ttnn::SmallVector<int32_t> begins(op->begins()->begin(),
                                      op->begins()->end());
  ::ttnn::SmallVector<int32_t> ends(op->ends()->begin(), op->ends()->end());
  ::ttnn::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ::ttnn::Tensor out = ::ttnn::slice(in, begins, ends, step);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
