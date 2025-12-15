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
  // INVALID - copy source to output tensor and apply no reduction(pass
  // std::nullopt)
  std::optional<std::string> scatterReduceTypeString;
  switch (op->scatter_reduce_type()) {
  case ::tt::target::ttnn::ScatterReduceType::Sum: // Sum -> add
    scatterReduceTypeString = "add";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Prod: // Prod -> multiply
    scatterReduceTypeString = "multiply";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Min: // Min -> amin
    scatterReduceTypeString = "amin";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Max: // Max -> amax
    scatterReduceTypeString = "amax";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Invalid: // Invalid -> no
                                                       // reduction
    scatterReduceTypeString = std::nullopt;
    break;
  }

  ::ttnn::Tensor out =
      ::ttnn::scatter(input, dim, index, source, outputMemoryConfig,
                      scatterReduceTypeString, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
