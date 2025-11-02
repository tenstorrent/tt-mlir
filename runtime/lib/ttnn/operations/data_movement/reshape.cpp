// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  if (shape.size() == 3 && shape[0] == 1 && shape[1] == 512 &&
      shape[2] == 768) {
    int x = 2;
    (void)x;
  }
  ::ttnn::Tensor out = ::ttnn::reshape(in, shape, memoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
