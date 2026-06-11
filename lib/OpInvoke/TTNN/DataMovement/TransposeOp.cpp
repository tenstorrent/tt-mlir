// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/TransposeOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

TransposeResolvedParams
resolveTransposeParams(const ::tt::target::ttnn::TransposeOpT &transposeOp) {
  TransposeResolvedParams params;

  if (transposeOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*transposeOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*transposeOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createTransposeTuple(Tag tag,
                          const ::tt::target::ttnn::TransposeOpT &transposeOp,
                          TensorArg input,
                          const TransposeResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), transposeOp.dim0,
                         transposeOp.dim1, params.outputMemoryConfig);
}

TransposeOpResult
callTranspose(CallType callType,
              const ::tt::target::ttnn::TransposeOpT &transposeOp,
              TensorArg input, ::ttnn::MeshDevice *device) {
  TransposeResolvedParams params = resolveTransposeParams(transposeOp);

  auto makeTuple = [&](auto tag) {
    return createTransposeTuple(tag, transposeOp, input, params);
  };

  return callOp<TransposeOpResult>(WRAP_OP(::ttnn::transpose), callType,
                                   makeTuple, device);
}

} // namespace ttnn_op_invoke
