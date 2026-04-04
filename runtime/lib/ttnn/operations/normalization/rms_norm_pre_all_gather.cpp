// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/rms_norm_pre_all_gather.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"

namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather {
void run(const ::tt::target::ttnn::RMSNormPreAllGatherOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> residual = std::nullopt;
  if (op->residual()) {
    residual = tensorPool.getTTNNTensorAndValidate(op->residual());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::prim::LayerNormProgramConfig programConfig;
  if (op->program_config()) {
    programConfig = utils::createLayerNormShardedMultiCoreProgramConfig(
        op->program_config());
  }

  bool use2DCoreGrid = op->use_2d_core_grid();

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  auto shardSpec = input.shard_spec();
  LOG_ASSERT(shardSpec.has_value(),
             "Input tensor must have shard spec for rms_norm_pre_all_gather");

  // Call TTNN RMS Norm Pre all-gather Op
  ::ttnn::Tensor output = ::ttnn::rms_norm_pre_all_gather(
      input, dtype, residual, computeConfig, programConfig, memoryConfig,
      use2DCoreGrid);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather
