// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction {
static void runReductionOp(
    const ::tt::target::ttnn::ReductionOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &,
        const std::optional<std::variant<int, ::ttsl::SmallVector<int>>> &,
        const bool, const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::DeviceComputeKernelConfig> &, float, bool,
        const std::optional<::ttnn::CoreRangeSet> &)> &ttnnOp) {

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbDimArg = op->dim_arg();
  std::optional<::ttsl::SmallVector<int>> dimArg =
      fbDimArg ? std::make_optional(::ttsl::SmallVector<int>(fbDimArg->begin(),
                                                             fbDimArg->end()))
               : std::nullopt;

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::Tensor out = ttnnOp(
      in, dimArg, op->keep_dim(), outputMemoryConfig /* memory_config_arg */,
      computeConfig /* compute_kernel_config */, 1.0f /* scalar */,
      true /* correction */, std::nullopt /* sub_core_grids */);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
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
  case ::tt::target::ttnn::ReductionOpType::Min: {
    runReductionOp(op, tensorPool, ::ttnn::min);
    break;
  }
  }
}
} // namespace tt::runtime::ttnn::operations::reduction
