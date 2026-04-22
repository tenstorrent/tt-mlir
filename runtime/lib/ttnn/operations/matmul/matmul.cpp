// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "matmul/unifiedMatmulOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "utils/utils.h"

#include <algorithm>
#include <optional>

namespace tt::runtime::ttnn::operations::matmul {

// ANCHOR: adding_an_op_matmul_runtime_operations
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->b());

  target::ttnn::MatmulOpT matmulOpT;
  op->UnPackTo(&matmulOpT);

  unifiedOpLib::MatmulOpResult result = unifiedOpLib::callMatmul(
      unifiedOpLib::CallType::EXECUTE, matmulOpT, &lhs, &rhs);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callMatmul execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

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

  target::ttnn::LinearOpT linearOpT;
  op->UnPackTo(&linearOpT);

  unifiedOpLib::LinearOpResult result = unifiedOpLib::callLinear(
      unifiedOpLib::CallType::EXECUTE, linearOpT, &lhs, &rhs,
      bias.has_value() ? std::make_optional(&bias.value()) : std::nullopt);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callLinear execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::SparseMatmulOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &a = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &b = tensorPool.getTTNNTensorAndValidate(op->b());
  const ::ttnn::Tensor &sparsity =
      tensorPool.getTTNNTensorAndValidate(op->sparsity());

  target::ttnn::SparseMatmulOpT sparseMatmulOpT;
  op->UnPackTo(&sparseMatmulOpT);

  //   auto outputMemoryConfig =
  //       ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
  //           ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  //   LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
  //                  outputMemoryConfig,
  //              "Memory config must exist for device tensors");

  //   std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  //   if (op->compute_config()) {
  //     computeConfig =
  //         utils::createDeviceComputeKernelConfig(op->compute_config());
  //   }

  //   std::optional<uint32_t> nnz =
  //       op->nnz() != 0 ? std::make_optional(static_cast<uint32_t>(op->nnz()))
  //                      : std::nullopt;

  //   // Read program config from the flatbuffer (populated at compile time by
  //   // TTIRToTTNN lowering).
  //   LOG_ASSERT(op->program_config(),
  //              "SparseMatmulOp requires program_config to be set at compile "
  //              "time");
  //   auto *config = op->program_config();
  //   target::ttnn::CoreRangeSetT hopCoresT;
  //   (*config->hop_cores()).UnPackTo(&hopCoresT);
  //   ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig
  //       programConfig{
  //           .compute_with_storage_grid_size =
  //               unifiedOpLib::operations::utils::toTTNNCoreCoord(
  //                   *config->compute_with_storage_grid_size()),
  //           .in0_block_w = config->in0_block_w(),
  //           .out_subblock_h = config->out_subblock_h(),
  //           .out_subblock_w = config->out_subblock_w(),
  //           .out_block_h = config->out_block_h(),
  //           .out_block_w = config->out_block_w(),
  //           .per_core_M = config->per_core_m(),
  //           .per_core_N = config->per_core_n(),
  //           .fuse_batch = config->fuse_batch(),
  //           .fused_activation =
  //               config->fused_activation()
  //                   ?
  //                   std::optional<::ttnn::operations::unary::UnaryWithParam>(
  //                         utils::toTTNNUnaryWithParam(
  //                             *config->fused_activation()))
  //                   : std::nullopt,
  //           .mcast_in0 = config->mcast_in0(),
  //           .gather_in0 = config->gather_in0(),
  //           .hop_cores =
  //               //
  //               ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*config->hop_cores()),
  //           unifiedOpLib::operations::utils::toTTNNCoreRangeSet(hopCoresT),
  //           .num_global_cb_receivers = config->num_global_cb_receivers(),
  //           .untilize_out = config->untilize_out()};

  unifiedOpLib::LinearOpResult result = unifiedOpLib::callSparseMatmul(
      unifiedOpLib::CallType::EXECUTE, sparseMatmulOpT, &a, &b, &sparsity);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callSparseMatmul execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  //   ::ttnn::Tensor output =
  //       ::ttnn::sparse_matmul(a, b, sparsity,
  //                             /*program_config=*/programConfig,
  //                             /*nnz=*/nnz,
  //                             /*is_input_a_sparse=*/op->is_input_a_sparse(),
  //                             /*is_input_b_sparse=*/op->is_input_b_sparse(),
  //                             /*memory_config=*/outputMemoryConfig,
  //                             /*dtype=*/std::nullopt,
  //                             /*compute_kernel_config=*/computeConfig,
  //                             /*core_grid=*/std::nullopt,
  //                             /*output_tile=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::matmul
