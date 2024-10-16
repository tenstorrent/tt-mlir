// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

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
  ::ttnn::Tensor out = ::ttnn::operations::matmul::matmul(
      lhs, rhs, /*bias=*/std::nullopt,
      ::ttnn::operations::matmul::Matmul{/*program_config=*/std::nullopt,
                                         /*bcast_batch=*/std::nullopt,
                                         outputMemoryConfig, outputDataType});
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::matmul
// ANCHOR_END: adding_an_op_matmul_runtime_operations
