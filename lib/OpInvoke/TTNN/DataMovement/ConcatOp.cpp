// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/ConcatOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ConcatResolvedParams
resolveConcatParams(const ::tt::target::ttnn::ConcatOpT &concatOp) {
  ConcatResolvedParams params;

  if (concatOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*concatOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*concatOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
inline auto resolveTensorArgVector(const std::vector<TensorArg> &args,
                                   Tag callType) {
  using ElemT = decltype(resolveTensorArg(std::declval<TensorArg>(), callType));
  std::vector<ElemT> result;
  for (const auto &arg : args) {
    result.push_back(resolveTensorArg(arg, callType));
  }
  return result;
}

template <typename Tag>
auto createConcatTuple(Tag tag, const ::tt::target::ttnn::ConcatOpT &concatOp,
                       const std::vector<TensorArg> &inputs,
                       const ConcatResolvedParams &params) {
  return std::make_tuple(resolveTensorArgVector(inputs, tag), concatOp.dim,
                         params.outputMemoryConfig);
}

ConcatOpResult callConcat(CallType callType,
                          const ::tt::target::ttnn::ConcatOpT &concatOp,
                          std::vector<TensorArg> inputs,
                          ::ttnn::MeshDevice *device) {
  ConcatResolvedParams params = resolveConcatParams(concatOp);

  auto makeTuple = [&](auto tag) {
    return createConcatTuple(tag, concatOp, inputs, params);
  };

  return callOp<ConcatOpResult>(WRAP_OP(::ttnn::concat), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
