// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/GatherOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

GatherResolvedParams
resolveGatherParams(const ::tt::target::ttnn::GatherOpT &gatherOp) {
  GatherResolvedParams params;

  params.dim = static_cast<int8_t>(gatherOp.dim);

  if (gatherOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*gatherOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*gatherOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createGatherTuple(Tag tag, const ::tt::target::ttnn::GatherOpT &gatherOp,
                       TensorArg input, TensorArg index,
                       const GatherResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.dim,
                         resolveTensorArg(index, tag),
                         /*sparse_grad=*/false, params.outputMemoryConfig);
}

GatherOpResult callGather(CallType callType,
                          const ::tt::target::ttnn::GatherOpT &gatherOp,
                          TensorArg input, TensorArg index,
                          ::ttnn::MeshDevice *device) {
  GatherResolvedParams params = resolveGatherParams(gatherOp);

  auto makeTuple = [&](auto tag) {
    return createGatherTuple(tag, gatherOp, input, index, params);
  };

  return callOp<GatherOpResult>(WRAP_OP(::ttnn::gather), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
