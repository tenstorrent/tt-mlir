// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::matmul {
// ANCHOR: adding_an_op_matmul_runtime
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  // Matmul args
  const bool transposeA = false;
  const bool transposeB = false;
  const std::optional<const ::tt::tt_metal::MemoryConfig> memoryConfig =
      std::make_optional(outputMemoryConfig);
  const std::optional<const ::ttnn::DataType> dtype =
      std::make_optional(outputDataType);
  const std::optional<const ::ttnn::operations::matmul::MatmulProgramConfig>
      programConfig = std::nullopt;
  const std::optional<const std::string> activation = std::nullopt;
  const std::optional<const ::ttnn::DeviceComputeKernelConfig>
      computeKernelConfig = std::nullopt;
  const std::optional<const ::ttnn::CoreGrid> coreGrid = std::nullopt;

  ::ttnn::Tensor out =
      ::ttnn::matmul(lhs, rhs, transposeA, transposeB, memoryConfig, dtype,
                     programConfig, activation, computeKernelConfig, coreGrid);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
// ANCHOR_END: adding_an_op_matmul_runtime

} // namespace tt::runtime::ttnn::operations::matmul
