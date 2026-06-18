// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/embedding/embedding_backward.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Embedding/EmbeddingBackwardOp.h"

namespace tt::runtime::ttnn::operations::embedding_backward {
void run(const ::tt::target::ttnn::EmbeddingBackwardOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &inGrad =
      tensorPool.getTTNNTensorAndValidate(op->in_grad());

  ::tt::target::ttnn::EmbeddingBackwardOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::EmbeddingBackwardOpResult result =
      ttnn_op_invoke::callEmbeddingBackward(ttnn_op_invoke::CallType::EXECUTE,
                                            opNative, &input, &weight, &inGrad,
                                            &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callEmbeddingBackward execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::embedding_backward
