// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/softmax.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/SoftmaxOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::normalization {
void run(const ::tt::target::ttnn::SoftmaxOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::SoftmaxOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::SoftmaxOpResult result = ttnn_op_invoke::callSoftmax(
      ttnn_op_invoke::CallType::EXECUTE, opT, &in, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callSoftmax execution");

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::normalization
