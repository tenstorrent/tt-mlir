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

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  const auto *fbDimArg = op->dim_arg();
  int dim = fbDimArg ? static_cast<int>(*fbDimArg->begin()) : 0;
  bool all_dimensions = fbDimArg ? false : true;

  ::ttnn::Tensor out = ::ttnn::prod(in, all_dimensions, dim, op->keep_dim(),
                                    outputMemoryConfig /* memory_config_arg */);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::ReductionProdOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionProdOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction
