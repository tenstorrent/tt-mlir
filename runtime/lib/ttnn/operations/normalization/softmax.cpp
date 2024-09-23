// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::normalization {
void run(const ::tt::target::ttnn::SoftmaxOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  int32_t dimension = op->dimension();
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());
  ::ttnn::Tensor out = ::ttnn::softmax(in, dimension, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::normalization
