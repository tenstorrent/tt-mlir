// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include <optional>

namespace tt::runtime::ttnn::operations::reduction {
static void
runReductionArgMaxOp(::tt::target::ttnn::ReductionArgMaxOp const *op,
                     ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  ::ttnn::Tensor out = ::ttnn::argmax(in, op->dim(), op->use_multicore(),
                                      /*memory_config_arg=*/outputMemoryConfig,
                                      /*optional_output_tensor=*/std::nullopt);

  tensorPool.insertAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::ReductionArgMaxOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionArgMaxOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction
