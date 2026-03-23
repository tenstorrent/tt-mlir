// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice_write.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SliceWriteOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &operand = tensorPool.getTTNNTensorAndValidate(op->operand());
  const ::ttnn::Tensor &starts =
      tensorPool.getTTNNTensorAndValidate(op->starts());

  // Read starts tensor from device to host to extract values.
  ::ttnn::Tensor startsHost = starts.cpu();
  const auto &startsBuffer =
      ::tt::tt_metal::host_buffer::get_host_buffer(startsHost);
  const auto &startsView = startsBuffer.view_as<int32_t>();

  auto inputShape = input.logical_shape();
  uint32_t rank = inputShape.rank();

  ::ttnn::SmallVector<uint32_t> begins(rank);
  ::ttnn::SmallVector<uint32_t> ends(rank);
  ::ttnn::SmallVector<uint32_t> step(rank, 1);

  for (uint32_t i = 0; i < rank; ++i) {
    begins[i] = static_cast<uint32_t>(startsView[i]);
    ends[i] = begins[i] + static_cast<uint32_t>(inputShape[i]);
  }

  ::ttnn::experimental::slice_write(input, operand, begins, ends, step);
}
} // namespace tt::runtime::ttnn::operations::data_movement
