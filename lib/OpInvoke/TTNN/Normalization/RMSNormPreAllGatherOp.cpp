// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/RMSNormPreAllGatherOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

RMSNormPreAllGatherResolvedParams resolveRMSNormPreAllGatherParams(
    const ::tt::target::ttnn::RMSNormPreAllGatherOpT &opT) {
  RMSNormPreAllGatherResolvedParams params;
  if (opT.out) {
    params.dtype = operations::utils::getDataType(*opT.out);
  }
  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
  }
  if (opT.program_config) {
    params.programConfig =
        operations::utils::createLayerNormShardedMultiCoreProgramConfig(
            *opT.program_config);
  }
  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  params.use2DCoreGrid = std::make_optional(opT.use_2d_core_grid);
  return params;
}

template <typename Tag>
auto createRMSNormPreAllGatherTuple(
    Tag tag, const ::tt::target::ttnn::RMSNormPreAllGatherOpT & /*opT*/,
    TensorArg input, std::optional<TensorArg> residual,
    const RMSNormPreAllGatherResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), params.dtype,
      residual ? std::make_optional(resolveTensorArg(*residual, tag))
               : std::nullopt,
      params.computeConfig, params.programConfig, params.outputMemoryConfig,
      params.use2DCoreGrid);
}

RMSNormPreAllGatherOpResult
callRMSNormPreAllGather(CallType callType,
                        const ::tt::target::ttnn::RMSNormPreAllGatherOpT &opT,
                        TensorArg input, std::optional<TensorArg> residual,
                        ::ttnn::MeshDevice *device) {
  RMSNormPreAllGatherResolvedParams params =
      resolveRMSNormPreAllGatherParams(opT);

  auto makeTuple = [&](auto tag) {
    return createRMSNormPreAllGatherTuple(tag, opT, input, residual, params);
  };

  return callOp<RMSNormPreAllGatherOpResult>(::ttnn::rms_norm_pre_all_gather,
                                             callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
