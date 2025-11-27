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
  // TT Metal currently only supports the following scatter reduction types:
  // SUM, MULTIPLY, AMIN, AMAX - applied reduction to source tensor
  // INVALID - copy source to output tensor
  ::ttnn::operations::data_movement::scatter::ScatterReductionType
      scatterReduceType;
  switch (op->scatter_reduce_type()) {
  case ::tt::target::ttnn::ScatterReduceType::Sum: // Sum -> ADD
    scatterReduceType =
        ::ttnn::operations::data_movement::scatter::ScatterReductionType::ADD;
    break;
  case ::tt::target::ttnn::ScatterReduceType::Prod: // Prod -> MULTIPLY
    scatterReduceType = ::ttnn::operations::data_movement::scatter::
        ScatterReductionType::MULTIPLY;
    break;
  case ::tt::target::ttnn::ScatterReduceType::Min: // Min -> AMIN
    scatterReduceType =
        ::ttnn::operations::data_movement::scatter::ScatterReductionType::AMIN;
    break;
  case ::tt::target::ttnn::ScatterReduceType::Max: // Max -> AMAX
    scatterReduceType =
        ::ttnn::operations::data_movement::scatter::ScatterReductionType::AMAX;
    break;
  case ::tt::target::ttnn::ScatterReduceType::Invalid: // Invalid -> INVALID
    scatterReduceType = ::ttnn::operations::data_movement::scatter::
        ScatterReductionType::INVALID;
    break;
  }
  // TODO(abogdanovic): Add scatterReduceType - std::string instead of enum
  ::ttnn::Tensor out =
      ::ttnn::scatter(input, dim, index, source, outputMemoryConfig,
                      std::nullopt, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
