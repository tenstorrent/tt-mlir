// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/layer_norm_pre_all_gather.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormPreAllGatherOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::layer_norm_pre_all_gather {
void run(const ::tt::target::ttnn::LayerNormPreAllGatherOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> residualInput = std::nullopt;
  if (op->residual_input()) {
    residualInput = tensorPool.getTTNNTensorAndValidate(op->residual_input());
  }

  std::optional<::ttnn::Tensor> recip = std::nullopt;
  if (op->recip()) {
    recip = tensorPool.getTTNNTensorAndValidate(op->recip());
  }

  ::tt::target::ttnn::LayerNormPreAllGatherOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::LayerNormPreAllGatherOpResult result =
      ttnn_op_invoke::callLayerNormPreAllGather(
          ttnn_op_invoke::CallType::EXECUTE, opT, &input,
          residualInput.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*residualInput)
              : std::nullopt,
          recip.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*recip)
                            : std::nullopt,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callLayerNormPreAllGather execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::layer_norm_pre_all_gather
