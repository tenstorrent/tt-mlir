// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/prepare_conv_transpose2d_weights.h"

#include "conv/unifiedPrepareConvTranspose2dWeightsOp.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConvTranspose2dWeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &weightTensor =
      tensorPool.getTTNNTensorAndValidate(op->weight_tensor());

  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  unifiedOpLib::PrepareConvTranspose2dWeightsOpResult result =
      unifiedOpLib::callPrepareConvTranspose2dWeights(
          unifiedOpLib::CallType::EXECUTE, opT, &weightTensor, targetDevice);

  LOG_ASSERT(
      std::holds_alternative<::ttnn::Tensor>(result),
      "Expected Tensor from callPrepareConvTranspose2dWeights execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::conv
