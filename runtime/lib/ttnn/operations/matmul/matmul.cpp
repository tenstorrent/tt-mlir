// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include <optional>

// ANCHOR: adding_an_op_matmul_runtime_operations
namespace tt::runtime::ttnn::operations::matmul {
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  DEBUG_ASSERT(lhs.is_allocated());
  DEBUG_ASSERT(rhs.is_allocated());
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  std::optional<
      ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>
      programConfig = std::nullopt;

  const std::optional<const ::tt::tt_metal::MemoryConfig> memoryConfig =
      std::make_optional(outputMemoryConfig);

  const std::optional<const ::ttnn::DataType> dtype =
      std::make_optional(outputDataType);

  ::ttnn::Tensor out = ::ttnn::matmul(
      lhs, rhs, /*transposeA*/ false, /*transposeB*/ false, memoryConfig, dtype,
      /*programConfig*/ std::nullopt, /*activation*/ std::nullopt,
      /*computeKernelConfig*/ std::nullopt, /*coreGrid*/ std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
  std::cout<<"lhs of matmul logical shape: "<<lhs.get_logical_shape()<<std::endl;
  std::cout<<"rhs of matmul logical shape: "<<rhs.get_logical_shape()<<std::endl;
  std::cout<<"output of matmul logical shape: "<<out.get_logical_shape()<<std::endl;
}
} // namespace tt::runtime::ttnn::operations::matmul
// ANCHOR_END: adding_an_op_matmul_runtime_operations
