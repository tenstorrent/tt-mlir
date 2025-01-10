// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <optional>

namespace tt::runtime::ttnn::operations::matmul {
// ANCHOR: adding_an_op_matmul_runtime_operations
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->a()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->b()->global_id());
  const ::ttnn::Tensor &out = tensorPool.at(op->out()->global_id());
  DEBUG_ASSERT(lhs.is_allocated());
  DEBUG_ASSERT(rhs.is_allocated());
  DEBUG_ASSERT(out.is_allocated());

  ::ttnn::matmul(lhs, rhs, op->transpose_a(), op->transpose_b(),
                 /*memory_config=*/std::nullopt, /*dtype=*/std::nullopt,
                 /*program_config=*/std::nullopt, /*activation=*/std::nullopt,
                 /*compute_kernel_config=*/std::nullopt,
                 /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt, out);
}
// ANCHOR_END: adding_an_op_matmul_runtime_operations

void run(const ::tt::target::ttnn::LinearOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;

  DEBUG_ASSERT(lhs.is_allocated());
  DEBUG_ASSERT(rhs.is_allocated());
  DEBUG_ASSERT(!bias || bias->is_allocated());

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  const std::optional<const ::ttnn::DataType> dtype =
      std::make_optional(outputDataType);

  ::ttnn::Tensor out = ::ttnn::linear(
      lhs, rhs, bias, /*transposeA*/ false, /*transposeB*/ false,
      outputMemoryConfig, dtype, /*programConfig*/ std::nullopt,
      /*activation*/ std::nullopt,
      /*computeKernelConfig*/ std::nullopt, /*coreGrid*/ std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::matmul
