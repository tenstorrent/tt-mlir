// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SliceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttsl::SmallVector<int32_t> begins(op->begins()->begin(),
                                      op->begins()->end());
  ::ttsl::SmallVector<int32_t> ends(op->ends()->begin(), op->ends()->end());
  ::ttsl::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ::ttsl::Span<const int32_t> beginsSpan = ::ttsl::make_const_span(begins);
  ::ttsl::Span<const int32_t> endsSpan = ::ttsl::make_const_span(ends);
  ::ttsl::Span<const int32_t> stepSpan = ::ttsl::make_const_span(step);

  ::ttnn::Tensor out = ::ttnn::slice(in, beginsSpan, endsSpan, stepSpan);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
