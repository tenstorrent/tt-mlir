// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/RMSNormPreAllGatherOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

RMSNormPreAllGatherResolvedParams resolveRMSNormPreAllGatherParams(
    const ::tt::target::ttnn::RMSNormPreAllGatherOpT &op) {
  RMSNormPreAllGatherResolvedParams params;
  if (op.out) {
    params.dtype = operations::utils::getDataType(*op.out);
  }
  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }
  if (op.program_config) {
    params.programConfig =
        operations::utils::createLayerNormShardedMultiCoreProgramConfig(
            *op.program_config);
  }
  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }
  params.use2DCoreGrid = std::make_optional(op.use_2d_core_grid);
  return params;
}

template <typename Tag>
auto createRMSNormPreAllGatherTuple(
    Tag tag, const ::tt::target::ttnn::RMSNormPreAllGatherOpT & /*op*/,
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
                        const ::tt::target::ttnn::RMSNormPreAllGatherOpT &op,
                        TensorArg input, std::optional<TensorArg> residual,
                        ::ttnn::MeshDevice *device) {
  RMSNormPreAllGatherResolvedParams params =
      resolveRMSNormPreAllGatherParams(op);

  auto makeTuple = [&](auto tag) {
    return createRMSNormPreAllGatherTuple(tag, op, input, residual, params);
  };

  return callOp<RMSNormPreAllGatherOpResult>(
      WRAP_OP(::ttnn::rms_norm_pre_all_gather), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
