// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduction.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::reduction {
static void runReductionOp(
    ::tt::target::ttnn::ReductionOp const *op, ProgramTensorPool &tensorPool,
    std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &,
        const std::optional<std::variant<int, std::vector<int>>> &, const bool,
        const std::optional<::tt::tt_metal::MemoryConfig> &,
        const std::optional<::ttnn::DeviceComputeKernelConfig> &, float)>
        ttnnOp) {
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());

  const auto *fbDimArg = op->dim_arg();
  std::optional<vector<int>> dimArg =
      fbDimArg ? std::make_optional(
                     std::vector<int>(fbDimArg->begin(), fbDimArg->end()))
               : std::nullopt;

  ::ttnn::Tensor out = ttnnOp(
      in, dimArg, op->keep_dim(), outputMemoryConfig /* memory_config_arg */,
      std::nullopt /* compute_kernel_config */, 1.0f /* scalar */);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::ReductionOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    runReductionOp(op, tensorPool, ::ttnn::sum);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    runReductionOp(op, tensorPool, ::ttnn::mean);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Max: {
    runReductionOp(op, tensorPool, ::ttnn::max);
    break;
  }
  }
}
} // namespace tt::runtime::ttnn::operations::reduction
