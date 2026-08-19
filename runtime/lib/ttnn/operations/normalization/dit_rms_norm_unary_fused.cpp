// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/dit_rms_norm_unary_fused.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace tt::runtime::ttnn::operations::dit_rms_norm_unary_fused {
void run(const ::tt::target::ttnn::DitRMSNormUnaryFusedOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  // Handle optional weight, bias, and residual operands.
  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<::ttnn::Tensor> residual = std::nullopt;
  if (op->residual_input()) {
    residual = tensorPool.getTTNNTensorAndValidate(op->residual_input());
  }

  float epsilon = op->epsilon();

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  // Convert the optional activation string (e.g. "silu", "gelu") into a
  // UnaryWithParam, matching the ttnn Python binding behavior.
  std::optional<::ttnn::operations::unary::UnaryWithParam> activation =
      std::nullopt;
  if (op->activation()) {
    activation = ::ttnn::operations::unary::utils::string_to_unary_with_param(
        op->activation()->str());
  }

  ::ttnn::Tensor output = ::ttnn::experimental::dit_rms_norm_unary_fused(
      input, epsilon, weight, bias, residual, memoryConfig,
      /*program_config=*/std::nullopt, computeConfig, activation);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::dit_rms_norm_unary_fused
