// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/embedding/embedding_backward.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::runtime::ttnn::operations::embedding_backward {
void run(const ::tt::target::ttnn::EmbeddingBackwardOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  const ::ttnn::Tensor &inGrad = tensorPool.at(op->in_grad()->global_id());
  DEBUG_ASSERT(input.is_allocated());
  DEBUG_ASSERT(weight.is_allocated());
  DEBUG_ASSERT(inGrad.is_allocated());

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }
  if (op->memcfg()) {
    memoryConfig =
        std::make_optional(utils::createMemoryConfig(op->memcfg(), op->out()));
  }
  ::ttnn::Tensor out =
      ::ttnn::embedding_bw(input, weight, inGrad, dtype, memoryConfig);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::embedding_backward
