// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/SoftmaxOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/softmax/softmax.hpp"

#include <optional>

namespace ttnn_op_invoke {

SoftmaxResolvedParams
resolveSoftmaxParams(const ::tt::target::ttnn::SoftmaxOpT &op) {
  SoftmaxResolvedParams params;
  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
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
auto createSoftmaxTuple(Tag tag, const ::tt::target::ttnn::SoftmaxOpT &op,
                        TensorArg input, const SoftmaxResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), op.dimension,
                         params.outputMemoryConfig, params.computeConfig,
                         op.numeric_stable);
}

SoftmaxOpResult callSoftmax(CallType callType,
                            const ::tt::target::ttnn::SoftmaxOpT &op,
                            TensorArg input, ::ttnn::MeshDevice *device) {
  SoftmaxResolvedParams params = resolveSoftmaxParams(op);

  auto makeTuple = [&](auto tag) {
    return createSoftmaxTuple(tag, op, input, params);
  };

  return callOp<SoftmaxOpResult>(WRAP_OP(::ttnn::softmax), callType, makeTuple,
                                 device);
}

} // namespace ttnn_op_invoke
