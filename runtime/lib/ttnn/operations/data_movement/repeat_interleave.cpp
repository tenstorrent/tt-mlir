// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/repeat.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include <ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp>
#include <ttnn/tensor/types.hpp>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::RepeatInterleaveOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  uint32_t repeats = op->repeats();
  int32_t dim = op->dim();
  MemoryConfig memory_config = op->output_mem_config()
                                   ? op->output_mem_config().value()
                                   : input.memory_config();
  ::ttnn::Tensor out = ::ttnn::repeat_interleave(input, repeats, dim);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
