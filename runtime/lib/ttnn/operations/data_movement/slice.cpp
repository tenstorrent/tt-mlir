// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SliceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttnn::SmallVector<int32_t> begins(op->begins()->begin(),
                                      op->begins()->end());
  ::ttnn::SmallVector<int32_t> ends(op->ends()->begin(), op->ends()->end());
  ::ttnn::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ::ttnn::Tensor out = ::ttnn::slice(in, begins, ends, step);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
