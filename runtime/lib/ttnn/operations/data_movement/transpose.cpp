// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/transpose.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::TransposeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());
  int32_t dim0 = op->dim0();
  int32_t dim1 = op->dim1();
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  ::ttnn::Tensor out = ::ttnn::transpose(in, dim0, dim1, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
