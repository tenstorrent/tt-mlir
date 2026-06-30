// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/prepare_conv_transpose2d_bias.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConvTranspose2dBiasOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConvTranspose2dBiasOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &biasTensor =
      tensorPool.getTTNNTensorAndValidate(op->bias_tensor());

  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
      prepareConvTranspose2dBiasOpNative;
  op->UnPackTo(&prepareConvTranspose2dBiasOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::PrepareConvTranspose2dBiasOpResult result =
      ttnn_op_invoke::callPrepareConvTranspose2dBias(
          ttnn_op_invoke::CallType::EXECUTE, prepareConvTranspose2dBiasOpNative,
          &biasTensor, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callPrepareConvTranspose2dBias execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::conv
