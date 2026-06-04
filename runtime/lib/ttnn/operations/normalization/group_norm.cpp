// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/group_norm.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/GroupNormOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::group_norm {
void run(const ::tt::target::ttnn::GroupNormOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> input_mask = std::nullopt;
  if (op->input_mask()) {
    input_mask = tensorPool.getTTNNTensorAndValidate(op->input_mask());
  }

  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  ::tt::target::ttnn::GroupNormOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::GroupNormOpResult result = ttnn_op_invoke::callGroupNorm(
      ttnn_op_invoke::CallType::EXECUTE, opT, &input,
      input_mask.has_value()
          ? std::optional<ttnn_op_invoke::TensorArg>(&*input_mask)
          : std::nullopt,
      weight.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*weight)
                         : std::nullopt,
      bias.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*bias)
                       : std::nullopt,
      &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callGroupNorm execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::group_norm
