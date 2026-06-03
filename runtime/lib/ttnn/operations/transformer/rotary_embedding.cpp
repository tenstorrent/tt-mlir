// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/rotary_embedding.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "ttmlir/OpInvoke/TTNN/transformer/rotaryEmbeddingOp.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void runRotaryEmbedding(const ::tt::target::ttnn::RotaryEmbeddingOp *op,
                               ProgramTensorPool &tensorPool,
                               ProgramContext &context) {
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &cosCache =
      tensorPool.getTTNNTensorAndValidate(op->cos_cache());
  const ::ttnn::Tensor &sinCache =
      tensorPool.getTTNNTensorAndValidate(op->sin_cache());

  ::tt::target::ttnn::RotaryEmbeddingOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::RotaryEmbeddingOpResult result =
      ttnn_op_invoke::callRotaryEmbedding(ttnn_op_invoke::CallType::EXECUTE,
                                          opT, &input, &cosCache, &sinCache,
                                          targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callRotaryEmbedding execution");
  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::RotaryEmbeddingOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runRotaryEmbedding(op, tensorPool, context);
}
} // namespace tt::runtime::ttnn::operations::transformer
