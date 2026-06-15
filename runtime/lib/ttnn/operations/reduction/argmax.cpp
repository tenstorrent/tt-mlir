// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/ArgMaxOp.h"

namespace tt::runtime::ttnn::operations::reduction {
static void
runReductionArgMaxOp(const ::tt::target::ttnn::ReductionArgMaxOp *op,
                     ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::ReductionArgMaxOpT reductionArgMaxOpNative;
  op->UnPackTo(&reductionArgMaxOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ArgMaxOpResult result =
      ttnn_op_invoke::callArgMax(ttnn_op_invoke::CallType::EXECUTE,
                                 reductionArgMaxOpNative, &in, &targetDevice);
  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callArgMax execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::ReductionArgMaxOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionArgMaxOp(op, tensorPool, context);
}
} // namespace tt::runtime::ttnn::operations::reduction
