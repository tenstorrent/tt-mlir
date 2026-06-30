// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/topk.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/TopKOp.h"

namespace tt::runtime::ttnn::operations::reduction::topk {
static void runReductionTopKOp(const ::tt::target::ttnn::TopKOp *op,
                               ProgramTensorPool &tensorPool,
                               ProgramContext &context) {
  const ::ttnn::Tensor &in =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());

  ::tt::target::ttnn::TopKOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::TopKOpResult result = ttnn_op_invoke::callTopK(
      ttnn_op_invoke::CallType::EXECUTE, opNative, &in, &targetDevice);

  LOG_ASSERT(std::holds_alternative<std::vector<::ttnn::Tensor>>(result),
             "Expected vector of Tensors from callTopK execution");

  const std::vector<::ttnn::Tensor> &outputs =
      std::get<std::vector<::ttnn::Tensor>>(result);

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}

void run(const ::tt::target::ttnn::TopKOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionTopKOp(op, tensorPool, context);
}
} // namespace tt::runtime::ttnn::operations::reduction::topk
