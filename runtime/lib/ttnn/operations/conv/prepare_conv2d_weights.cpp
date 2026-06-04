// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/prepare_conv2d_weights.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConv2dWeightsOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::conv {

void run(const ::tt::target::ttnn::PrepareConv2dWeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &weightTensor =
      tensorPool.getTTNNTensorAndValidate(op->weight_tensor());

  ::tt::target::ttnn::PrepareConv2dWeightsOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::PrepareConv2dWeightsOpResult result =
      ttnn_op_invoke::callPrepareConv2dWeights(
          ttnn_op_invoke::CallType::EXECUTE, opT, &weightTensor, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callPrepareConv2dWeights execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::conv
