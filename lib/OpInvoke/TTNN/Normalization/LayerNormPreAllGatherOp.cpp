// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormPreAllGatherOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_pre_all_gather.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

LayerNormPreAllGatherResolvedParams resolveLayerNormPreAllGatherParams(
    const ::tt::target::ttnn::LayerNormPreAllGatherOpT &opT) {
  LayerNormPreAllGatherResolvedParams params;
  if (opT.out) {
    params.dtype = operations::utils::getDataType(*opT.out);
  } else {
    params.dtype = ttnn::DataType::BFLOAT16;
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
auto createLayerNormPreAllGatherTuple(
    Tag tag, const ::tt::target::ttnn::LayerNormPreAllGatherOpT & /*op*/,
    TensorArg input, std::optional<TensorArg> residualInput,
    std::optional<TensorArg> recip,
    const LayerNormPreAllGatherResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), params.dtype,
      residualInput ? std::make_optional(resolveTensorArg(*residualInput, tag))
                    : std::nullopt,
      params.computeConfig, params.programConfig, params.outputMemoryConfig,
      recip ? std::make_optional(resolveTensorArg(*recip, tag)) : std::nullopt);
}

LayerNormPreAllGatherOpResult callLayerNormPreAllGather(
    CallType callType, const ::tt::target::ttnn::LayerNormPreAllGatherOpT &opT,
    TensorArg input, std::optional<TensorArg> residualInput,
    std::optional<TensorArg> recip, ::ttnn::MeshDevice *device) {
  LayerNormPreAllGatherResolvedParams params =
      resolveLayerNormPreAllGatherParams(opT);

  auto makeTuple = [&](auto tag) {
    return createLayerNormPreAllGatherTuple(tag, opT, input, residualInput,
                                            recip, params);
  };

  return callOp<LayerNormPreAllGatherOpResult>(
      ::ttnn::layer_norm_pre_all_gather, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
