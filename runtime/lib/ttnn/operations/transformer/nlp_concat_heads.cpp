// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/nlp_concat_heads.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runNLPConcatHeadsOp(const ::tt::target::ttnn::NLPConcatHeadsOp *op,
                                ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::Tensor out =
      ::ttnn::experimental::nlp_concat_heads(in, outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::NLPConcatHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runNLPConcatHeadsOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
