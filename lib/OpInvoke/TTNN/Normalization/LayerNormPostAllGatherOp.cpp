// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormPostAllGatherOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_post_all_gather.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

LayerNormPostAllGatherResolvedParams resolveLayerNormPostAllGatherParams(
    const ::tt::target::ttnn::LayerNormPostAllGatherOpT &opT) {
  LayerNormPostAllGatherResolvedParams params;
  if (opT.out) {
    params.outputDtype = operations::utils::getDataType(*opT.out);
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
  return params;
}

template <typename Tag>
auto createLayerNormPostAllGatherTuple(
    Tag tag, const ::tt::target::ttnn::LayerNormPostAllGatherOpT &opT,
    TensorArg input, TensorArg stats, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias,
    const LayerNormPostAllGatherResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(stats, tag), opT.epsilon,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      params.outputMemoryConfig, params.computeConfig, params.programConfig,
      params.outputDtype);
}

LayerNormPostAllGatherOpResult callLayerNormPostAllGather(
    CallType callType, const ::tt::target::ttnn::LayerNormPostAllGatherOpT &opT,
    TensorArg input, TensorArg stats, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  LayerNormPostAllGatherResolvedParams params =
      resolveLayerNormPostAllGatherParams(opT);

  auto makeTuple = [&](auto tag) {
    return createLayerNormPostAllGatherTuple(tag, opT, input, stats, weight,
                                             bias, params);
  };

  return callOp<LayerNormPostAllGatherOpResult>(
      ::ttnn::layer_norm_post_all_gather, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
