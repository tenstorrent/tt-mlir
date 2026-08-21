// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/layer_norm_pre_all_gather.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_pre_all_gather.hpp"

namespace tt::runtime::ttnn::operations::layer_norm_pre_all_gather {
namespace {

// Metal Welford fatals on Float32 unless fp32_dest_acc_en is true. Match the
// RMSNorm high-precision default when the compiler omitted a config.
::ttnn::DeviceComputeKernelConfig defaultDistributedLayerNormComputeConfig() {
  ::ttnn::WormholeComputeKernelConfig config;
  config.math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4;
  config.math_approx_mode = false;
  config.fp32_dest_acc_en = true;
  config.packer_l1_acc = true;
  return config;
}

} // namespace

void run(const ::tt::target::ttnn::LayerNormPreAllGatherOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> residualInput = std::nullopt;
  if (op->residual_input()) {
    residualInput = tensorPool.getTTNNTensorAndValidate(op->residual_input());
  }

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  std::optional<const ::ttnn::DeviceComputeKernelConfig> computeConfig =
      utils::toMetalDeviceComputeKernelConfig(op->compute_config());
  if (!computeConfig.has_value()) {
    computeConfig.emplace(defaultDistributedLayerNormComputeConfig());
  }

  std::optional<::ttnn::prim::LayerNormProgramConfig> programConfig =
      std::nullopt;
  if (op->program_config()) {
    programConfig = utils::createLayerNormShardedMultiCoreProgramConfig(
        op->program_config());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::Tensor> recip = std::nullopt;
  if (op->recip()) {
    recip = tensorPool.getTTNNTensorAndValidate(op->recip());
  }

  ::ttnn::Tensor output = ::ttnn::layer_norm_pre_all_gather(
      input, dtype, residualInput, computeConfig, programConfig, memoryConfig,
      recip, /*fast_and_approximate_mode=*/false);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::layer_norm_pre_all_gather
