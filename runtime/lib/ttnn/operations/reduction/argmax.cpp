// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <optional>

namespace tt::runtime::ttnn::operations::reduction {
static void
runReductionArgMaxOp(const ::tt::target::ttnn::ReductionArgMaxOp *op,
                     ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  ::ttnn::Tensor out = ::ttnn::argmax(in, op->dim(),
                                      /*keepdim=*/op->keep_dim(),
                                      /*sub_core_grids=*/std::nullopt,
                                      /*use_multicore=*/op->use_multicore(),
                                      /*memory_config_arg=*/outputMemoryConfig,
                                      /*optional_output_tensor=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::ReductionArgMaxOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionArgMaxOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction
