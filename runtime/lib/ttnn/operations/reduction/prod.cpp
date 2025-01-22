// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/prod.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction {
static void runReductionProdOp(::tt::target::ttnn::ReductionProdOp const *op,
                               ProgramTensorPool &tensorPool) {

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      op->memcfg() ? std::make_optional(
                         utils::createMemoryConfig(op->memcfg(), op->out()))
                   : std::nullopt;

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  ::ttnn::Tensor out =
      ::ttnn::prod(in, op->all_dimensions(), op->dim_arg(), op->keep_dim(),
                   outputMemoryConfig /* memory_config_arg */);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::ReductionProdOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionProdOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction
