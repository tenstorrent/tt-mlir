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
    const ::tt::target::ttnn::LayerNormPreAllGatherOpT &op) {
  LayerNormPreAllGatherResolvedParams params;
  if (op.out) {
    params.dtype = operations::utils::getDataType(*op.out);
  } else {
    params.dtype = ttnn::DataType::BFLOAT16;
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
    LOG_ASSERT(operations::utils::inSystemMemory(*op.out) ||
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
    CallType callType, const ::tt::target::ttnn::LayerNormPreAllGatherOpT &op,
    TensorArg input, std::optional<TensorArg> residualInput,
    std::optional<TensorArg> recip, ::ttnn::MeshDevice *device) {
  LayerNormPreAllGatherResolvedParams params =
      resolveLayerNormPreAllGatherParams(op);

  auto makeTuple = [&](auto tag) {
    return createLayerNormPreAllGatherTuple(tag, op, input, residualInput,
                                            recip, params);
  };

  return callOp<LayerNormPreAllGatherOpResult>(
      WRAP_OP(::ttnn::layer_norm_pre_all_gather), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
