// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/topk.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction::topk {
static void runReductionTopKOp(const ::tt::target::ttnn::TopKOp *op,
                               ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::topk(in, op->k(), op->dim(),
                   /*largest=*/op->largest(),
                   /*sorted=*/op->sorted(),
                   /*memory_config=*/outputMemoryConfig,
                   /*sub_core_grids=*/std::nullopt,
                   /*indices_tensor=*/std::nullopt,
                   /*preallocated_output_tensors=*/std::nullopt);

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}

void run(const ::tt::target::ttnn::TopKOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionTopKOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction::topk
