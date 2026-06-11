// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormPostAllGatherOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_post_all_gather.hpp"

#include <optional>

namespace ttnn_op_invoke {

LayerNormPostAllGatherResolvedParams resolveLayerNormPostAllGatherParams(
    const ::tt::target::ttnn::LayerNormPostAllGatherOpT &op) {
  LayerNormPostAllGatherResolvedParams params;
  if (op.out) {
    params.outputDtype = operations::utils::getDataType(*op.out);
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
  return params;
}

template <typename Tag>
auto createLayerNormPostAllGatherTuple(
    Tag tag, const ::tt::target::ttnn::LayerNormPostAllGatherOpT &op,
    TensorArg input, TensorArg stats, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias,
    const LayerNormPostAllGatherResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(stats, tag), op.epsilon,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      params.outputMemoryConfig, params.computeConfig, params.programConfig,
      params.outputDtype);
}

LayerNormPostAllGatherOpResult callLayerNormPostAllGather(
    CallType callType, const ::tt::target::ttnn::LayerNormPostAllGatherOpT &op,
    TensorArg input, TensorArg stats, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  LayerNormPostAllGatherResolvedParams params =
      resolveLayerNormPostAllGatherParams(op);

  auto makeTuple = [&](auto tag) {
    return createLayerNormPostAllGatherTuple(tag, op, input, stats, weight,
                                             bias, params);
  };

  return callOp<LayerNormPostAllGatherOpResult>(
      WRAP_OP(::ttnn::layer_norm_post_all_gather), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
