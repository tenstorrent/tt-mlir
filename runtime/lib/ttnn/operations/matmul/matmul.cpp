// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <operations/core/compute_kernel/compute_kernel_config.hpp>
#include <optional>

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

  auto config = ::ttnn::WormholeComputeKernelConfig();
  config.fp32_dest_acc_en = true;
  ::ttnn::Tensor output = ::ttnn::matmul(
      lhs, rhs, op->transpose_a(), op->transpose_b(), outputMemoryConfig,
      outputDataType, matmulProgramConfig,
      /*activation=*/std::nullopt, /*compute_kernel_config=*/config,
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
  auto config = ::ttnn::WormholeComputeKernelConfig();
  config.fp32_dest_acc_en = true;
  ::ttnn::Tensor output = ::ttnn::linear(
      lhs, rhs, bias, op->transpose_a(), op->transpose_b(), outputMemoryConfig,
      outputDataType, /*program_config=*/std::nullopt,
      /*activation=*/std::nullopt, /*compute_kernel_config=*/config,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /* optional_output_tensor=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::matmul
