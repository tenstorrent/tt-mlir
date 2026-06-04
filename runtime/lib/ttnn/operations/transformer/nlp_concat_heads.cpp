// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/nlp_concat_heads.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/NLPConcatHeadsOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void runNLPConcatHeadsOp(const ::tt::target::ttnn::NLPConcatHeadsOp *op,
                                ProgramTensorPool &tensorPool,
                                ProgramContext &context) {
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::NLPConcatHeadsOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::NLPConcatHeadsOpResult result =
      ttnn_op_invoke::callNLPConcatHeads(ttnn_op_invoke::CallType::EXECUTE, opT,
                                         &input, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callNLPConcatHeads execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::NLPConcatHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runNLPConcatHeadsOp(op, tensorPool, context);
}

} // namespace tt::runtime::ttnn::operations::transformer
