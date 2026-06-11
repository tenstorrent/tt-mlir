// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv_transpose2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Conv/ConvTranspose2dOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::ConvTranspose2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  ::tt::target::ttnn::ConvTranspose2dOpT convTranspose2dOpT;
  op->UnPackTo(&convTranspose2dOpT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ConvTranspose2dOpResult result =
      ttnn_op_invoke::callConvTranspose2d(
          ttnn_op_invoke::CallType::EXECUTE, convTranspose2dOpT, &input,
          &weight,
          bias.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*bias)
                           : std::nullopt,
          &targetDevice);

  LOG_ASSERT(
      std::holds_alternative<::ttnn::ConvTranspose2dResultWithOptions>(result),
      "Expected ConvTranspose2dResultWithOptions from callConvTranspose2d "
      "execution");

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(
                 std::get<::ttnn::ConvTranspose2dResultWithOptions>(result)),
             "Expected output Tensor in ConvTranspose2dResultWithOptions");

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(
      std::get<::ttnn::ConvTranspose2dResultWithOptions>(result));

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::conv
