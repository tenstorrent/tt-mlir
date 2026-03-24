// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice_write.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttnn/operations/experimental/slice_write/slice_write.hpp"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::SliceWriteOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &operand = tensorPool.getTTNNTensorAndValidate(op->operand());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &beginsTensor =
      tensorPool.getTTNNTensorAndValidate(op->begins());
  const ::ttnn::Tensor &endsTensor =
      tensorPool.getTTNNTensorAndValidate(op->ends());

  // Read begins and ends from device to host as int32 vectors.
  auto beginsVec = beginsTensor.to_vector<int32_t>();
  auto endsVec = endsTensor.to_vector<int32_t>();

  ttsl::SmallVector<uint32_t> begins(beginsVec.begin(), beginsVec.end());
  ttsl::SmallVector<uint32_t> ends(endsVec.begin(), endsVec.end());
  ttsl::SmallVector<uint32_t> step(begins.size(), 1);

  ::ttnn::experimental::slice_write(input, operand, begins, ends, step);

  // slice_write is in-place on operand, no need to insert a new tensor.
}

} // namespace tt::runtime::ttnn::operations::data_movement
