// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "ttmlir/OpInvoke/TTNN/conv/conv2dOp.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <variant>

namespace tt::runtime::ttnn::operations::conv {
using ::ttnn::Conv2dResultWithOptions;

void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  ::tt::target::ttnn::Conv2dOpT conv2dOpT;
  op->UnPackTo(&conv2dOpT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::Conv2dOpResult result = ttnn_op_invoke::callConv2d(
      ttnn_op_invoke::CallType::EXECUTE, conv2dOpT, &input, &weight,
      bias.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*bias)
                       : std::nullopt,
      targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Conv2dResultWithOptions>(result),
             "Expected Conv2dResultWithOptions from callConv2d execution");

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(
                 std::get<::ttnn::Conv2dResultWithOptions>(result)),
             "Expected output Tensor in Conv2dResultWithOptions");

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(
      std::get<::ttnn::Conv2dResultWithOptions>(result));

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
