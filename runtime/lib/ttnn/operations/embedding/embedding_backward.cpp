// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/embedding/embedding_backward.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::runtime::ttnn::operations::embedding_backward {
void run(const ::tt::target::ttnn::EmbeddingBackwardOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getAndValidate(op->input());
  const ::ttnn::Tensor &weight = tensorPool.getAndValidate(op->weight());
  const ::ttnn::Tensor &inGrad = tensorPool.getAndValidate(op->in_grad());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }
  ::ttnn::Tensor out =
      ::ttnn::embedding_bw(input, weight, inGrad, dtype, memoryConfig);

  tensorPool.insertAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::embedding_backward
