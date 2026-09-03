// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <algorithm>
#include <optional>
#include "ttnn/operations/experimental/quasar/matmul/matmul.hpp"

namespace tt::runtime::ttnn::operations::matmul {

static bool programCarriesFusedActivation(
    const std::optional<::ttnn::operations::matmul::MatmulProgramConfig> &pc) {
  if (!pc) {
    return false;
  }
  return std::visit(
      [](const auto &cfg) -> bool {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (
            std::is_same_v<T, ::ttnn::operations::matmul::
                                  MatmulMultiCoreReuseMultiCastProgramConfig> ||
            std::is_same_v<T,
                           ::ttnn::operations::matmul::
                               MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
            std::is_same_v<
                T, ::ttnn::operations::matmul::
                       MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> ||
            std::is_same_v<
                T,
                ::ttnn::operations::matmul::
                    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
          return cfg.fused_activation.has_value();
        }
        return false;
      },
      *pc);
}

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

  std::optional<std::string> activation;
  if (op->activation() && !programCarriesFusedActivation(matmulProgramConfig)) {
    activation = op->activation()->str();
  }

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

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  auto programConfig = utils::createMatmulProgramConfigIfNeeded(op);

  std::optional<std::string> activation;
  if (op->activation() && !programCarriesFusedActivation(programConfig)) {
    activation = op->activation()->str();
  }

  // Quasar's linear takes the same leading arguments; Activation is the same
  // std::variant<std::string, UnaryWithParam> alias in both headers, so the
  // std::optional<std::string> converts identically.
  // Quasar's matmul declares its OWN program-config types
  // (experimental::quasar::matmul::MatmulMultiCore*ProgramConfig), so the
  // mainline variant is not convertible and the two calls cannot share one
  // lambda. Forge does not currently emit a matmul program config, so passing
  // none is correct -- but refuse loudly if that ever changes rather than
  // silently dropping it, which would quietly alter the schedule.
  ::ttnn::Tensor output;
  if (utils::isQuasar()) {
    LOG_ASSERT(!programConfig.has_value(),
               "Quasar matmul: a mainline matmul program config cannot be "
               "translated to the Quasar program-config types. Add a "
               "translation before emitting one for Quasar.");
    output = ::ttnn::operations::experimental::quasar::matmul::linear(
        lhs, rhs, bias, op->transpose_a(), op->transpose_b(),
        outputMemoryConfig, outputDataType, /*program_config=*/std::nullopt,
        /*activation=*/activation, /*compute_kernel_config=*/computeConfig);
  } else {
    output = ::ttnn::linear(
        lhs, rhs, bias, op->transpose_a(), op->transpose_b(),
        outputMemoryConfig, outputDataType, programConfig,
        /*activation=*/activation, /*compute_kernel_config=*/computeConfig,
        /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

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
      op->nnz() ? std::make_optional(static_cast<uint32_t>(*op->nnz()))
                : std::nullopt;

  // Read program config from the flatbuffer (populated at compile time by
  // TTIRToTTNN lowering).
  LOG_ASSERT(op->program_config(),
             "SparseMatmulOp requires program_config to be set at compile "
             "time");
  auto *config = op->program_config();
  ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig
      programConfig{
          .compute_with_storage_grid_size =
              ::tt::runtime::ttnn::utils::toTTNNCoreCoord(
                  *config->compute_with_storage_grid_size()),
          .in0_block_w = config->in0_block_w(),
          .out_subblock_h = config->out_subblock_h(),
          .out_subblock_w = config->out_subblock_w(),
          .out_block_h = config->out_block_h(),
          .out_block_w = config->out_block_w(),
          .per_core_M = config->per_core_m(),
          .per_core_N = config->per_core_n(),
          .fuse_batch = config->fuse_batch(),
          .fused_activation =
              config->fused_activation()
                  ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                        utils::toTTNNUnaryWithParam(
                            *config->fused_activation()))
                  : std::nullopt,
          .mcast_in0 = config->mcast_in0(),
          .gather_in0 = config->gather_in0(),
          .hop_cores = ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(
              *config->hop_cores()),
          .num_global_cb_receivers = config->num_global_cb_receivers(),
          .untilize_out = config->untilize_out()};

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
