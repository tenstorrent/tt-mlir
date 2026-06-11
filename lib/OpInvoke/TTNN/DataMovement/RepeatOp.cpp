// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/RepeatOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

RepeatResolvedParams
resolveRepeatParams(const ::tt::target::ttnn::RepeatOpT &repeatOp) {
  RepeatResolvedParams params;

  params.repeatDims = {repeatOp.repeat_dims.begin(),
                       repeatOp.repeat_dims.end()};

  if (repeatOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*repeatOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*repeatOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createRepeatTuple(Tag tag, TensorArg input,
                       const RepeatResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.repeatDims,
                         params.outputMemoryConfig);
}

RepeatOpResult callRepeat(CallType callType,
                          const ::tt::target::ttnn::RepeatOpT &repeatOp,
                          TensorArg input, ::ttnn::MeshDevice *device) {
  RepeatResolvedParams params = resolveRepeatParams(repeatOp);

  auto makeTuple = [&](auto tag) {
    return createRepeatTuple(tag, input, params);
  };

  return callOp<RepeatOpResult>(WRAP_OP(::ttnn::repeat), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
