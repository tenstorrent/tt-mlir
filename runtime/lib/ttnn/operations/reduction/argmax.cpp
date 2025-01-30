// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/prod.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <optional>

namespace tt::runtime::ttnn::operations::reduction {
static void
runReductionArgMaxOp(::tt::target::ttnn::ReductionArgMaxOp const *op,
                     ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      op->memcfg() ? std::make_optional(
                         utils::createMemoryConfig(op->memcfg(), op->out()))
                   : std::nullopt;

  ::ttnn::Tensor out =
      ::ttnn::argmax(in, op->dim(), op->use_multicore(),
                     outputMemoryConfig /* memory_config_arg */, std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::ReductionArgMaxOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionArgMaxOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction
