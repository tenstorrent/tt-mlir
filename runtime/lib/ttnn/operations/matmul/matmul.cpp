// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <algorithm>
#include <iostream>
#include <optional>

// For operator<< on Tensor
#include "ttnn/tensor/tensor.hpp"
namespace tt::runtime::ttnn::operations::matmul {

// ANCHOR: adding_an_op_matmul_runtime_operations
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->b());

  auto outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig,
             "Memory config must exist for device tensors");

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulProgramConfig = utils::createMatmulProgramConfigIfNeeded(op);

  std::optional<std::string> activation =
      op->activation() ? std::make_optional(op->activation()->str())
                       : std::nullopt;

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::Tensor output = ::ttnn::matmul(
      lhs, rhs, op->transpose_a(), op->transpose_b(), outputMemoryConfig,
      outputDataType, matmulProgramConfig,
      /*activation=*/activation, /*compute_kernel_config=*/computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /* optional_output_tensor=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
// ANCHOR_END: adding_an_op_matmul_runtime_operations

void run(const ::tt::target::ttnn::LinearOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->b());
  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  auto outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig,
             "Memory config must exist for device tensors");

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  std::optional<std::string> activation =
      op->activation() ? std::make_optional(op->activation()->str())
                       : std::nullopt;

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  auto programConfig = utils::createMatmulProgramConfigIfNeeded(op);

  ::ttnn::Tensor output = ::ttnn::linear(
      lhs, rhs, bias, op->transpose_a(), op->transpose_b(), outputMemoryConfig,
      outputDataType, programConfig,
      /*activation=*/activation, /*compute_kernel_config=*/computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /* optional_output_tensor=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

namespace {
// Helper function to create program config for sparse_matmul
// Following GPT-OSS demo pattern for dynamic per_core_N calculation
::ttnn::operations::matmul::MatmulProgramConfig
createSparseMatmulProgramConfig(const ::ttnn::Tensor &inputA,
                                const ::ttnn::Tensor &inputB) {
  // Warmhole configureation
  constexpr uint32_t coreX = 8;
  constexpr uint32_t coreY = 8;
  constexpr uint32_t tileH = 32;
  constexpr uint32_t tileW = 32;
  const auto &shapeA = inputA.logical_shape();
  const auto &shapeB = inputB.logical_shape();

  // Get dimensions: A is [..., M, K], B is [..., K, N]
  uint32_t M = shapeA[-2];
  uint32_t N = shapeB[-1];

  // uint32_t MTiles = (M + tileH - 1) / tileH;
  uint32_t NTiles = (N + tileW - 1) / tileW;

  uint32_t perCoreM = std::max(1u, M / tileH);

  // For sparse_matmul, we need num_blocks_total to form a rectangular grid
  // that matches the core grid allocation.
  // num_blocks_y = ceil(MTiles / perCoreM)
  // num_blocks_x = ceil(NTiles / perCoreN)
  // num_blocks_total = num_blocks_y * num_blocks_x
  //
  // The simplest approach: set perCoreN so that num_blocks_x <= coreX
  // This ensures num_blocks_total <= coreX * num_blocks_y <= coreX * coreY

  // perCoreN = ceil(NTiles / coreX) ensures num_blocks_x <= coreX
  uint32_t perCoreN = std::max(1u, (NTiles + coreX - 1) / coreX);

  ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig
      config;
  config.compute_with_storage_grid_size = tt::tt_metal::CoreCoord(coreX, coreY);
  config.in0_block_w = 1;
  config.out_subblock_h = 1;
  config.out_subblock_w = 1;
  config.out_block_h = 1;
  config.out_block_w = 1;
  config.per_core_M = perCoreM;
  config.per_core_N = perCoreN;
  config.fuse_batch = false;
  config.fused_activation = std::nullopt;
  config.mcast_in0 = true;
  config.gather_in0 = false;
  config.num_global_cb_receivers = 0;
  config.untilize_out = false;
  return config;
}
} // namespace

void run(const ::tt::target::ttnn::SparseMatmulOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &a = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &b = tensorPool.getTTNNTensorAndValidate(op->b());
  const ::ttnn::Tensor &sparsity =
      tensorPool.getTTNNTensorAndValidate(op->sparsity());

  auto outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig,
             "Memory config must exist for device tensors");

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<uint32_t> nnz =
      op->nnz() != 0 ? std::make_optional(static_cast<uint32_t>(op->nnz()))
                     : std::nullopt;

  auto programConfig = createSparseMatmulProgramConfig(a, b);

  ::ttnn::Tensor output =
      ::ttnn::sparse_matmul(a, b, sparsity,
                            /*program_config=*/programConfig,
                            /*nnz=*/nnz,
                            /*is_input_a_sparse=*/op->is_input_a_sparse(),
                            /*is_input_b_sparse=*/op->is_input_b_sparse(),
                            /*memory_config=*/outputMemoryConfig,
                            /*dtype=*/std::nullopt,

                            /*compute_kernel_config=*/computeConfig,
                            /*core_grid=*/std::nullopt,
                            /*output_tile=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::matmul
