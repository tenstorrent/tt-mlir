// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/scatter.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::ScatterOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &index =
      tensorPool.getTTNNTensorAndValidate(op->index());
  const ::ttnn::Tensor &source =
      tensorPool.getTTNNTensorAndValidate(op->source());
  int32_t dim = op->dim();

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  // TT Metal currently only supports the following scatter reduction types: SUM, MULTIPLY, AMIN, AMAX.
  ::ttnn::operations::data_movement::scatter::ScatterReductionType scatter_reduce_type;
  switch (op->scatter_reduce_type()) {
  case 0: // Sum -> ADD
    scatter_reduce_type = ::ttnn::operations::data_movement::scatter::ScatterReductionType::ADD;
    break;
  case 6: // Mul -> MULTIPLY
    scatter_reduce_type = ::ttnn::operations::data_movement::scatter::ScatterReductionType::MULTIPLY;
    break;
  case 3: // Min -> AMIN
    scatter_reduce_type = ::ttnn::operations::data_movement::scatter::ScatterReductionType::AMIN;
    break;
  case 2: // Max -> AMAX
    scatter_reduce_type = ::ttnn::operations::data_movement::scatter::ScatterReductionType::AMAX;
    break;
  default:
    scatter_reduce_type = ::ttnn::operations::data_movement::scatter::ScatterReductionType::ADD;
    break;
  }

  ::ttnn::Tensor out = ::ttnn::scatter(input, dim, index, source,
                                       outputMemoryConfig, scatter_reduce_type);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
