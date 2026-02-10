// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/sort.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include <optional>
#include <vector>

namespace tt::runtime::ttnn::operations::data_movement {
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

  // Typecast outputs if the dtype produced by tt-metal differs from what the
  // MLIR IR expects. This is needed because tt-metal's sort autonomously
  // chooses UINT16 for index tensors, but the compiler may declare them as
  // a different type (e.g. si32).
  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    ::ttnn::DataType expectedDtype =
        ::tt::runtime::ttnn::operations::utils::getDataType(
            op->outputs()->Get(i));
    if (outputs[i].dtype() != expectedDtype) {
      outputs[i] =
          ::ttnn::typecast(outputs[i], expectedDtype, outputMemoryConfig);
    }
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::data_movement
