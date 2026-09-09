// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::ReductionOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

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

  ::ttnn::Tensor out;
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    out = ::ttnn::sum(in, dimArg, op->keep_dim(), outputMemoryConfig,
                      computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    out = ::ttnn::mean(in, dimArg, op->keep_dim(), outputMemoryConfig,
                       computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Max: {
    out = ::ttnn::max(in, dimArg, op->keep_dim(), outputMemoryConfig,
                      computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Min: {
    out = ::ttnn::min(in, dimArg, op->keep_dim(), outputMemoryConfig,
                      computeConfig);
    break;
  }
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::reduction
