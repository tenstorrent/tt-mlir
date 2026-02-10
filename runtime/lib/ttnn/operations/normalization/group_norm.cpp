// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/group_norm.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::group_norm {
void run(const ::tt::target::ttnn::GroupNormOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> input_mask = std::nullopt;
  if (op->input_mask()) {
    input_mask = tensorPool.getTTNNTensorAndValidate(op->input_mask());
  }

  // Handle optional weight and bias parameters
  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  float epsilon = op->epsilon();

  int num_groups = static_cast<int>(op->num_groups());

  // Handle optional memory config
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  // Get core_grid: use compile-time value from flatbuffer if present,
  // otherwise fallback to device grid size (required by ttnn::group_norm)
  const auto *coreGridFb = op->core_grid();
  auto grid_size = coreGridFb
      ? tt::tt_metal::CoreCoord(coreGridFb->x(), coreGridFb->y())
      : input.device()->compute_with_storage_grid_size();
  ::ttnn::CoreGrid core_grid(grid_size.x, grid_size.y);

  // Call TTNN group norm operation
  ::ttnn::Tensor output =
      ::ttnn::group_norm(input, num_groups, epsilon, input_mask, weight, bias,
                         /*reciprocals=*/std::nullopt, memoryConfig,
                         /*dtype=*/std::nullopt, core_grid);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::group_norm
