// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::matmul {
// ANCHOR: adding_an_op_matmul_runtime
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  // Matmul requires tile layout
  lhs = ::ttnn::to_layout(lhs, ::ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                          lhs.device());

  rhs = ::ttnn::to_layout(rhs, ::ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                          rhs.device());

  ::ttnn::Tensor out = ::ttnn::operations::matmul::matmul(
      lhs, rhs, /*bias=*/std::nullopt,
      ::ttnn::operations::matmul::Matmul{/*program_config=*/std::nullopt,
                                         /*bcast_batch=*/std::nullopt,
                                         outputMemoryConfig, outputDataType});
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
// ANCHOR_END: adding_an_op_matmul_runtime

} // namespace tt::runtime::ttnn::operations::matmul
