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

  // Use core_grid from flatbuffer when the compiler specified one; otherwise
  // pass nullopt so tt-metal can choose a valid grid (see tt-metal#40916).
  // Defaulting to the full device grid was invalid for large Ht*W (e.g. 8x8
  // when only 8x6 fits).
  const auto *coreGridFb = op->core_grid();
  std::optional<::ttnn::CoreGrid> coreGridOpt = std::nullopt;
  if (coreGridFb) {
    coreGridOpt.emplace(coreGridFb->x(), coreGridFb->y());
  }

  // Call TTNN group norm operation
  ::ttnn::Tensor output = ::ttnn::group_norm(
      input, num_groups, epsilon, input_mask, weight, bias,
      /*reciprocals=*/std::nullopt, memoryConfig,
      /*dtype=*/std::nullopt, coreGridOpt,
      /*inplace=*/std::nullopt, /*output_layout=*/std::nullopt,
      /*num_out_blocks=*/-1, /*compute_kernel_config=*/std::nullopt,
      /*negative_mask=*/std::nullopt,
      /*use_welford=*/false);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::group_norm
