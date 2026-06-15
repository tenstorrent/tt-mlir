// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/cumsum.h"

#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/CumSumOp.h"

namespace tt::runtime::ttnn::operations::reduction::cumsum {
void run(const ::tt::target::ttnn::CumSumOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::CumSumOpT cumSumOpNative;
  op->UnPackTo(&cumSumOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::CumSumOpResult result = ttnn_op_invoke::callCumSum(
      ttnn_op_invoke::CallType::EXECUTE, cumSumOpNative, &in, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callCumSum execution");

  const ::ttnn::Tensor &output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::reduction::cumsum
