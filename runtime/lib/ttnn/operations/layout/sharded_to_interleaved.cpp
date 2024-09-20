// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved.h"
#include "impl/buffers/buffer_constants.hpp"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.h"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::layout {

void run(const ::tt::target::ttnn::ShardedToInterleavedOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in0()->global_id());

  ::ttnn::MemoryConfig memcfg = inputTensor.memory_config();
  memcfg.memory_layout = TensorMemoryLayout::INTERLEAVED;
  ::ttnn::Tensor output = ::ttnn::sharded_to_interleaved(0, inputTensor, memcfg,
                                                         inputTensor.dtype());

  tensorPool.insert_or_assign(op->out()->global_id(), output);
}
} // namespace tt::runtime::ttnn::operations::layout
