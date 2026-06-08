// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/layer_norm.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::layer_norm {
void run(const ::tt::target::ttnn::LayerNormOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  // Handle optional weight and bias parameters
  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  ::tt::target::ttnn::LayerNormOpT layerNormOpNative;
  op->UnPackTo(&layerNormOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::LayerNormOpResult result = ttnn_op_invoke::callLayerNorm(
      ttnn_op_invoke::CallType::EXECUTE, layerNormOpNative, &input,
      weight.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*weight)
                         : std::nullopt,
      bias.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*bias)
                       : std::nullopt,
      &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callLayerNorm execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::layer_norm
