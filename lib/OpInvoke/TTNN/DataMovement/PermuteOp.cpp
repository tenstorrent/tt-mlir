// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/PermuteOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

PermuteResolvedParams
resolvePermuteParams(const ::tt::target::ttnn::PermuteOpT &permuteOp) {
  PermuteResolvedParams params;

  params.dims = {permuteOp.permutation.begin(), permuteOp.permutation.end()};

  if (permuteOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*permuteOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*permuteOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createPermuteTuple(Tag tag,
                        const ::tt::target::ttnn::PermuteOpT &permuteOp,
                        TensorArg input, const PermuteResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.dims,
                         params.outputMemoryConfig, permuteOp.pad_value);
}

PermuteOpResult callPermute(CallType callType,
                            const ::tt::target::ttnn::PermuteOpT &permuteOp,
                            TensorArg input, ::ttnn::MeshDevice *device) {
  PermuteResolvedParams params = resolvePermuteParams(permuteOp);

  auto makeTuple = [&](auto tag) {
    return createPermuteTuple(tag, permuteOp, input, params);
  };

  return callOp<PermuteOpResult>(WRAP_OP(::ttnn::permute), callType, makeTuple,
                                 device);
}

} // namespace ttnn_op_invoke
