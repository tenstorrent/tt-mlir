// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <optional>
#include <ttnn/operations/eltwise/unary/common/unary_op_types.hpp>
#include <ttnn/operations/matmul/device/matmul_op.hpp>

namespace tt::runtime::ttnn::operations::matmul {

::ttnn::operations::matmul::MatmulProgramConfig
createMatmulProgramConfig(const ::tt::target::ttnn::MatmulOp *op) {
  ::ttnn::operations::matmul::MatmulProgramConfig matmulProgramConfig;
  switch (op->matmul_program_config_type()) {
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseProgramConfig();
    matmulProgramConfig =
        ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size =
                {config->compute_with_storage_grid_size()->x(),
                 config->compute_with_storage_grid_size()->y()},
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
        };
    break;
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastProgramConfig();
    std::optional<::ttnn::operations::unary::UnaryWithParam> fused_activation;
    if (config->fused_activation()) {
      fused_activation = ::ttnn::operations::unary::UnaryWithParam(
          utils::toTTNNUnaryOpType(config->fused_activation()->op_type()),
          std::vector<float>(config->fused_activation()->params()->begin(),
                             config->fused_activation()->params()->end()));
    }

    matmulProgramConfig =
        ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size =
                {config->compute_with_storage_grid_size()->x(),
                 config->compute_with_storage_grid_size()->y()},
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .transpose_mcast = config->transpose_mcast(),
            .fused_activation = fused_activation,
            .fuse_batch = config->fuse_batch(),
        };
    break;
  }
  default:
    LOG_ERROR("Unsupported MatmulProgramConfig type");
    break;
  }
  return matmulProgramConfig;
}

// ANCHOR: adding_an_op_matmul_runtime_operations
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getAndValidate(op->b());

  auto outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig,
             "Memory config must exist for device tensors");

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulProgramConfig;
  if (op->matmul_program_config()) {
    matmulProgramConfig = createMatmulProgramConfig(op);
  }

  if (matmulProgramConfig.has_value()) {
    LOG_INFO("- MatmulProgramConfig is set");
    if (op->matmul_program_config_type() ==
        ::tt::target::ttnn::MatmulProgramConfig::
            MatmulMultiCoreReuseProgramConfig) {
      auto &config = std::get<
          ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(
          matmulProgramConfig.value());
      LOG_INFO("- Config type: MatmulMultiCoreReuseProgramConfig");
      auto message = fmt::format(
          "- MatmulMultiCoreReuseProgramConfig created with: grid_size=({}, "
          "{}), in0_block_w={}, out_subblock_h={}, out_subblock_w={}, "
          "per_core_M={}, per_core_N={}",
          config.compute_with_storage_grid_size.x,
          config.compute_with_storage_grid_size.y, config.in0_block_w,
          config.out_subblock_h, config.out_subblock_w, config.per_core_M,
          config.per_core_N);
      std::cout << message << std::endl;

    } else if (op->matmul_program_config_type() ==
               ::tt::target::ttnn::MatmulProgramConfig::
                   MatmulMultiCoreReuseMultiCastProgramConfig) {
      auto &config = std::get<::ttnn::operations::matmul::
                                  MatmulMultiCoreReuseMultiCastProgramConfig>(
          matmulProgramConfig.value());
      LOG_INFO("- Config type: MatmulMultiCoreReuseMultiCastProgramConfig");
      auto message = fmt::format(
          "- MatmulMultiCoreReuseMultiCastProgramConfig created with: "
          "grid_size=({}, {}), in0_block_w={}, out_subblock_h={}, "
          "out_subblock_w={}, out_block_h={}, out_block_w={}, per_core_M={}, "
          "per_core_N={}, transpose_mcast={}, has_fused_activation={}, "
          "fuse_batch={}",
          config.compute_with_storage_grid_size.x,
          config.compute_with_storage_grid_size.y, config.in0_block_w,
          config.out_subblock_h, config.out_subblock_w, config.out_block_h,
          config.out_block_w, config.per_core_M, config.per_core_N,
          config.transpose_mcast, config.fused_activation ? 1 : 0,
          config.fuse_batch);
      std::cout << message << std::endl;
    }
  } else {
    LOG_INFO("- MatmulProgramConfig is NOT set");
  }

  ::ttnn::Tensor output = ::ttnn::matmul(
      lhs, rhs, op->transpose_a(), op->transpose_b(), outputMemoryConfig,
      outputDataType, matmulProgramConfig,
      /*activation=*/std::nullopt, /*compute_kernel_config=*/std::nullopt,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /* optional_output_tensor=*/std::nullopt);

  tensorPool.insertAndValidate(op->out(), output);
}
// ANCHOR_END: adding_an_op_matmul_runtime_operations

void run(const ::tt::target::ttnn::LinearOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getAndValidate(op->b());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.getAndValidate(op->bias()))
                 : std::nullopt;

  auto outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig,
             "Memory config must exist for device tensors");

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  ::ttnn::Tensor output = ::ttnn::linear(
      lhs, rhs, bias, op->transpose_a(), op->transpose_b(), outputMemoryConfig,
      outputDataType, /*program_config=*/std::nullopt,
      /*activation=*/std::nullopt, /*compute_kernel_config=*/std::nullopt,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /* optional_output_tensor=*/std::nullopt);

  tensorPool.insertAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::matmul
