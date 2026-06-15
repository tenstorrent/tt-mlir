// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/prod.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/ProdOp.h"

namespace tt::runtime::ttnn::operations::reduction {
static void runReductionProdOp(const ::tt::target::ttnn::ReductionProdOp *op,
                               ProgramTensorPool &tensorPool,
                               ProgramContext &context) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::ReductionProdOpT prodOpNative;
  op->UnPackTo(&prodOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ProdOpResult result = ttnn_op_invoke::callProd(
      ttnn_op_invoke::CallType::EXECUTE, prodOpNative, &in, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callProd execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::ReductionProdOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionProdOp(op, tensorPool, context);
}
} // namespace tt::runtime::ttnn::operations::reduction
