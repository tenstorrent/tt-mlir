// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/layer_norm_post_all_gather.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/normalization/layernorm_distributed/layernorm_post_all_gather.hpp"

namespace tt::runtime::ttnn::operations::layer_norm_post_all_gather {
void run(const ::tt::target::ttnn::LayerNormPostAllGatherOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &stats = tensorPool.getTTNNTensorAndValidate(op->stats());

  float epsilon = op->epsilon();

  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<::ttnn::prim::LayerNormProgramConfig> programConfig =
      std::nullopt;
  if (op->program_config()) {
    programConfig = utils::createLayerNormShardedMultiCoreProgramConfig(
        op->program_config());
  }

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = unifiedOpLib::operations::utils::toTTNNDataType(*op->dtype());
  }

  ::ttnn::Tensor output = ::ttnn::layer_norm_post_all_gather(
      input, stats, epsilon, weight, bias, memoryConfig, computeConfig,
      programConfig, dtype);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::layer_norm_post_all_gather
