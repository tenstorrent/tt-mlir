// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/SoftmaxOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/softmax/softmax.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

SoftmaxResolvedParams
resolveSoftmaxParams(const ::tt::target::ttnn::SoftmaxOpT &opT) {
  SoftmaxResolvedParams params;
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
auto createSoftmaxTuple(Tag tag, const ::tt::target::ttnn::SoftmaxOpT &opT,
                        TensorArg input, const SoftmaxResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), opT.dimension,
                         params.outputMemoryConfig, params.computeConfig,
                         opT.numeric_stable);
}

SoftmaxOpResult callSoftmax(CallType callType,
                            const ::tt::target::ttnn::SoftmaxOpT &opT,
                            TensorArg input, ::ttnn::MeshDevice *device) {
  SoftmaxResolvedParams params = resolveSoftmaxParams(opT);

  auto makeTuple = [&](auto tag) {
    return createSoftmaxTuple(tag, opT, input, params);
  };

  return callOp<SoftmaxOpResult>(WRAP_OP(::ttnn::softmax), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
