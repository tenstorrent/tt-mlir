// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/ReshapeOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ReshapeResolvedParams
resolveReshapeParams(const ::tt::target::ttnn::ReshapeOpT &reshapeOp) {
  ReshapeResolvedParams params;

  params.shape = {reshapeOp.shape.begin(), reshapeOp.shape.end()};

  if (reshapeOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*reshapeOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*reshapeOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createReshapeTuple(Tag tag, TensorArg input,
                        const ReshapeResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.shape,
                         params.outputMemoryConfig);
}

ReshapeOpResult callReshape(CallType callType,
                            const ::tt::target::ttnn::ReshapeOpT &reshapeOp,
                            TensorArg input, ::ttnn::MeshDevice *device) {
  ReshapeResolvedParams params = resolveReshapeParams(reshapeOp);

  auto makeTuple = [&](auto tag) {
    return createReshapeTuple(tag, input, params);
  };

  return callOp<ReshapeOpResult>(WRAP_OP(::ttnn::reshape), callType, makeTuple,
                                 device);
}

} // namespace ttnn_op_invoke
