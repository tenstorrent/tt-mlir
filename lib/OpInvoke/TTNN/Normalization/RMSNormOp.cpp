// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/RMSNormOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

RMSNormResolvedParams
resolveRMSNormParams(const ::tt::target::ttnn::RMSNormOpT &opT) {
  RMSNormResolvedParams params;
  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
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
auto createRMSNormTuple(Tag tag, const ::tt::target::ttnn::RMSNormOpT &opT,
                        TensorArg input, std::optional<TensorArg> weight,
                        std::optional<TensorArg> bias,
                        const RMSNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), opT.epsilon,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      /*residual_input_tensor=*/std::nullopt, params.outputMemoryConfig,
      /*program_config=*/std::nullopt, params.computeConfig);
}

RMSNormOpResult callRMSNorm(CallType callType,
                            const ::tt::target::ttnn::RMSNormOpT &opT,
                            TensorArg input, std::optional<TensorArg> weight,
                            std::optional<TensorArg> bias,
                            ::ttnn::MeshDevice *device) {
  RMSNormResolvedParams params = resolveRMSNormParams(opT);

  auto makeTuple = [&](auto tag) {
    return createRMSNormTuple(tag, opT, input, weight, bias, params);
  };

  return callOp<RMSNormOpResult>(::ttnn::rms_norm, callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
