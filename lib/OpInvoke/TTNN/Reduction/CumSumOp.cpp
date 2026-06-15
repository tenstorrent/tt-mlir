// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Reduction/CumSumOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

CumSumResolvedParams
resolveCumSumParams(const ::tt::target::ttnn::CumSumOpT &cumSumOp) {
  CumSumResolvedParams params;

  if (cumSumOp.dtype.has_value()) {
    params.dtype = operations::utils::getDataType(*cumSumOp.out);
  }

  if (cumSumOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*cumSumOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*cumSumOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createCumSumTuple(Tag tag, const ::tt::target::ttnn::CumSumOpT &cumSumOp,
                       TensorArg input, const CumSumResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), cumSumOp.dim, params.dtype,
      /*reverse_order=*/false,
      /*optional_out=*/std::nullopt, params.outputMemoryConfig);
}

CumSumOpResult callCumSum(CallType callType,
                          const ::tt::target::ttnn::CumSumOpT &cumSumOp,
                          TensorArg input, ::ttnn::MeshDevice *device) {
  CumSumResolvedParams params = resolveCumSumParams(cumSumOp);

  auto makeTuple = [&](auto tag) {
    return createCumSumTuple(tag, cumSumOp, input, params);
  };

  return callOp<CumSumOpResult>(WRAP_OP(::ttnn::cumsum), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
