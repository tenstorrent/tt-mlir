// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pad.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PadOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  float padValue = op->value();

  std::array<uint32_t, 2> output_shape = {1};
  std::array<uint32_t, 2> input_shape = {1};

  std::copy_n(op->output_shape()->begin(), op->output_shape()->size(),
              output_shape.begin());
  std::copy_n(op->input_shape()->begin(), op->input_shape()->size(),
              input_shape.begin());

  ::ttnn::Tensor out = ::ttnn::pad(in, output_shape, input_shape, padValue);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
