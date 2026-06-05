// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

LayerNormResolvedParams
resolveLayerNormParams(const ::tt::target::ttnn::LayerNormOpT &opT) {
  LayerNormResolvedParams params;
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
auto createLayerNormTuple(Tag tag, const ::tt::target::ttnn::LayerNormOpT &opT,
                          TensorArg input, std::optional<TensorArg> weight,
                          std::optional<TensorArg> bias,
                          const LayerNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), opT.epsilon,
      weight ? std::make_optional(resolveTensorArg(*weight, tag)) : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      /*residual_input_tensor=*/std::nullopt, params.outputMemoryConfig,
      /*program_config=*/std::nullopt,
      /*compute_kernel_config=*/std::nullopt,
      /*recip_tensor=*/std::nullopt);
}

LayerNormOpResult
callLayerNorm(CallType callType, const ::tt::target::ttnn::LayerNormOpT &opT,
              TensorArg input, std::optional<TensorArg> weight,
              std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  LayerNormResolvedParams params = resolveLayerNormParams(opT);

  auto makeTuple = [&](auto tag) {
    return createLayerNormTuple(tag, opT, input, weight, bias, params);
  };

  callOp(::ttnn::layer_norm);
}

} // namespace ttnn_op_invoke
