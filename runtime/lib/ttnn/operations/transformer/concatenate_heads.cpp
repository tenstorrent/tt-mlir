// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/concatenate_heads.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/ConcatenateHeadsOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void
runConcatenateHeadsOp(const ::tt::target::ttnn::ConcatenateHeadsOp *op,
                      ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::ConcatenateHeadsOpT concatenateHeadsOpNative;
  op->UnPackTo(&concatenateHeadsOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ConcatenateHeadsOpResult result =
      ttnn_op_invoke::callConcatenateHeads(ttnn_op_invoke::CallType::EXECUTE,
                                           concatenateHeadsOpNative, &input,
                                           &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callConcatenateHeads execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::ConcatenateHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runConcatenateHeadsOp(op, tensorPool, context);
}

} // namespace tt::runtime::ttnn::operations::transformer
