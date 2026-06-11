// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/rotary_embedding_llama.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/RotaryEmbeddingLlamaOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void
runRotaryEmbeddingLlama(const ::tt::target::ttnn::RotaryEmbeddingLlamaOp *op,
                        ProgramTensorPool &tensorPool,
                        ProgramContext &context) {
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &cosCache =
      tensorPool.getTTNNTensorAndValidate(op->cos_cache());
  const ::ttnn::Tensor &sinCache =
      tensorPool.getTTNNTensorAndValidate(op->sin_cache());
  const ::ttnn::Tensor &transMat =
      tensorPool.getTTNNTensorAndValidate(op->trans_mat());

  ::tt::target::ttnn::RotaryEmbeddingLlamaOpT rotaryEmbeddingLlamaOpNative;
  op->UnPackTo(&rotaryEmbeddingLlamaOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::RotaryEmbeddingLlamaOpResult result =
      ttnn_op_invoke::callRotaryEmbeddingLlama(
          ttnn_op_invoke::CallType::EXECUTE, rotaryEmbeddingLlamaOpNative,
          &input, &cosCache, &sinCache, &transMat, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callRotaryEmbeddingLlama execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::RotaryEmbeddingLlamaOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runRotaryEmbeddingLlama(op, tensorPool, context);
}

} // namespace tt::runtime::ttnn::operations::transformer
