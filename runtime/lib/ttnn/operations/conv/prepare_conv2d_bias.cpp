// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/prepare_conv2d_bias.h"

#include "conv/unifiedPrepareConv2dBiasOp.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConv2dBiasOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &biasTensor =
      tensorPool.getTTNNTensorAndValidate(op->bias_tensor());

  ::tt::target::ttnn::PrepareConv2dBiasOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  unifiedOpLib::PrepareConv2dBiasOpResult result =
      unifiedOpLib::callPrepareConv2dBias(unifiedOpLib::CallType::EXECUTE, opT,
                                          &biasTensor, targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callPrepareConv2dBias execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::conv
