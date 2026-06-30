// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Reduction/ProdOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ProdResolvedParams
resolveProdParams(const ::tt::target::ttnn::ReductionProdOpT &prodOp) {
  ProdResolvedParams params;

  if (prodOp.dim_arg.has_value()) {
    params.dimArg = *prodOp.dim_arg;
  }

  if (prodOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*prodOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*prodOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createProdTuple(Tag tag,
                     const ::tt::target::ttnn::ReductionProdOpT &prodOp,
                     TensorArg input, const ProdResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.dimArg,
                         prodOp.keep_dim, params.outputMemoryConfig);
}

ProdOpResult callProd(CallType callType,
                      const ::tt::target::ttnn::ReductionProdOpT &prodOp,
                      TensorArg input, ::ttnn::MeshDevice *device) {
  ProdResolvedParams params = resolveProdParams(prodOp);

  auto makeTuple = [&](auto tag) {
    return createProdTuple(tag, prodOp, input, params);
  };

  return callOp<ProdOpResult, true, false>(WRAP_OP(::ttnn::prod), callType,
                                           makeTuple, device);
}

} // namespace ttnn_op_invoke
