// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/sort.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include <optional>
#include <vector>

namespace tt::runtime::ttnn::operations::sort {
void run(const ::tt::target::ttnn::SortOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::sort(in, op->dim(), op->descending(), op->stable(),
                   outputMemoryConfig, std::nullopt);

  LOG_ASSERT(
      op->outputs()->size() == outputs.size(),
      "Number of expected outputs does not match with generated outputs.");
  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::sort
