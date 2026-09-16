// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"

#include <variant>
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttnn/operations/experimental/quasar/reduction/generic/generic_reductions.hpp"

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

  // Quasar's reduction entry points take the dim as a variant rather than a plain
  // SmallVector, so the same value has to be rebuilt in that shape.
  std::optional<std::variant<int, int64_t, ::ttsl::SmallVector<int>>> quasarDimArg;
  if (dimArg.has_value()) {
    quasarDimArg = *dimArg;
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::Tensor out;
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    out = utils::isQuasar()
              ? ::ttnn::operations::experimental::quasar::sum(
                    in, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                    computeConfig)
              : ::ttnn::sum(in, dimArg, op->keep_dim(), outputMemoryConfig,
                      computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    out = utils::isQuasar()
              ? ::ttnn::operations::experimental::quasar::mean(
                    in, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                    computeConfig)
              : ::ttnn::mean(in, dimArg, op->keep_dim(), outputMemoryConfig,
                       computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Max: {
    out = utils::isQuasar()
              ? ::ttnn::operations::experimental::quasar::max(
                    in, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                    computeConfig)
              : ::ttnn::max(in, dimArg, op->keep_dim(), outputMemoryConfig,
                      computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Min: {
    out = utils::isQuasar()
              ? ::ttnn::operations::experimental::quasar::min(
                    in, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                    computeConfig)
              : ::ttnn::min(in, dimArg, op->keep_dim(), outputMemoryConfig,
                      computeConfig);
    break;
  }
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::reduction
