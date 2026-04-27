// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv_transpose2d.h"
#include "conv/unifiedConvTranspose2dOp.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::ConvTranspose2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<unifiedOpLib::TensorArg> bias;
  if (op->bias()) {
    const ::ttnn::Tensor &biasTensor =
        tensorPool.getTTNNTensorAndValidate(op->bias());
    bias = &biasTensor;
  }

  ::tt::target::ttnn::ConvTranspose2dOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  unifiedOpLib::ConvTranspose2dOpResult result =
      unifiedOpLib::callConvTranspose2d(unifiedOpLib::CallType::EXECUTE, opT,
                                        &input, &weight, bias, targetDevice);

  LOG_ASSERT(
      std::holds_alternative<::ttnn::ConvTranspose2dResultWithOptions>(result),
      "Expected ConvTranspose2dResultWithOptions from "
      "callConvTranspose2d execution");

  auto &convResult = std::get<::ttnn::ConvTranspose2dResultWithOptions>(result);
  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(convResult),
             "Expected Tensor result from conv_transpose2d");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(convResult);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::conv
