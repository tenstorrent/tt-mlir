// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/embedding/embedding.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::embedding {
void run(const ::tt::target::ttnn::EmbeddingOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  DEBUG_ASSERT(input.is_allocated());
  DEBUG_ASSERT(weight.is_allocated());

  // default params for embedding op
  std::optional<int> padToken = std::nullopt;
  ::ttnn::Layout layout = utils::isTilized(op->out())
                              ? ::ttnn::TILE_LAYOUT
                              : ::ttnn::ROW_MAJOR_LAYOUT;
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ::ttnn::embedding(input, weight, padToken, layout, embeddingsType,
                        outputDataType, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::embedding
