// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Reduction/ArgMaxOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ArgMaxResolvedParams
resolveArgMaxParams(const ::tt::target::ttnn::ReductionArgMaxOpT &argMaxOp) {
  ArgMaxResolvedParams params;

  if (argMaxOp.dim.has_value()) {
    params.dim = *argMaxOp.dim;
  }

  if (argMaxOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*argMaxOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*argMaxOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createArgMaxTuple(Tag tag,
                       const ::tt::target::ttnn::ReductionArgMaxOpT &argMaxOp,
                       TensorArg input, const ArgMaxResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), params.dim, argMaxOp.keep_dim,
      /*sub_core_grids=*/std::nullopt, params.outputMemoryConfig,
      /*optional_output_tensor=*/std::nullopt);
}

ArgMaxOpResult
callArgMax(CallType callType,
           const ::tt::target::ttnn::ReductionArgMaxOpT &argMaxOp,
           TensorArg input, ::ttnn::MeshDevice *device) {
  ArgMaxResolvedParams params = resolveArgMaxParams(argMaxOp);

  auto makeTuple = [&](auto tag) {
    return createArgMaxTuple(tag, argMaxOp, input, params);
  };

  return callOp<ArgMaxOpResult>(WRAP_OP(::ttnn::argmax), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
