// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/RepeatInterleaveOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

RepeatInterleaveResolvedParams resolveRepeatInterleaveParams(
    const ::tt::target::ttnn::RepeatInterleaveOpT &repeatInterleaveOp) {
  RepeatInterleaveResolvedParams params;

  if (repeatInterleaveOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*repeatInterleaveOp.out));
    TT_INVOKE_ASSERT(
        operations::utils::inSystemMemory(*repeatInterleaveOp.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createRepeatInterleaveTuple(
    Tag tag, const ::tt::target::ttnn::RepeatInterleaveOpT &repeatInterleaveOp,
    TensorArg input, const RepeatInterleaveResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         repeatInterleaveOp.repeats, repeatInterleaveOp.dim,
                         params.outputMemoryConfig);
}

RepeatInterleaveOpResult callRepeatInterleave(
    CallType callType,
    const ::tt::target::ttnn::RepeatInterleaveOpT &repeatInterleaveOp,
    TensorArg input, ::ttnn::MeshDevice *device) {
  RepeatInterleaveResolvedParams params =
      resolveRepeatInterleaveParams(repeatInterleaveOp);

  auto makeTuple = [&](auto tag) {
    return createRepeatInterleaveTuple(tag, repeatInterleaveOp, input, params);
  };

  return callOp<RepeatInterleaveOpResult>(WRAP_OP(::ttnn::repeat_interleave),
                                          callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
