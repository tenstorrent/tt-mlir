// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/rms_norm.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::rms_norm {
void run(const ::tt::target::ttnn::RMSNormOp *op, ProgramContext &context) {
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

  float epsilon = op->epsilon();

  // Handle optional memory config
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  // Call TTNN RMS norm operation
  ::ttnn::Tensor output = ::ttnn::rms_norm(
      input, epsilon, weight, bias,
      /*residual_input_tensor=*/std::nullopt, // Not used in our implementation
      memoryConfig,
      /*program_config=*/std::nullopt,       // Use default
      /*compute_kernel_config=*/std::nullopt // Use default
  );

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::rms_norm
