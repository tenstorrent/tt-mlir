// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::where {

void run(const ::tt::target::ttnn::WhereOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &pred = tensorPool.at(op->pred()->global_id());
  const ::ttnn::Tensor &on_true = tensorPool.at(op->on_true()->global_id());
  const ::ttnn::Tensor &on_false = tensorPool.at(op->on_false()->global_id());

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->output());

  ::ttnn::Tensor out =
      ::ttnn::where(pred, on_true, on_false, outputMemoryConfig);
  tensorPool.insert_or_assign(op->output()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::where
