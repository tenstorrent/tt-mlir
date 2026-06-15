// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Reduction/ReductionOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ReductionResolvedParams
resolveReductionParams(const ::tt::target::ttnn::ReductionOpT &reductionOp) {
  ReductionResolvedParams params;

  if (!reductionOp.dim_arg.empty()) {
    params.dimArg = ::ttsl::SmallVector<int>(reductionOp.dim_arg.begin(),
                                             reductionOp.dim_arg.end());
  }

  if (reductionOp.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *reductionOp.compute_config);
  }

  if (reductionOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*reductionOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*reductionOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createReductionTuple(Tag tag,
                          const ::tt::target::ttnn::ReductionOpT &reductionOp,
                          TensorArg input,
                          const ReductionResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.dimArg,
                         reductionOp.keep_dim, params.outputMemoryConfig,
                         params.computeConfig,
                         /*scalar=*/1.0f,
                         /*correction=*/true,
                         /*sub_core_grids=*/std::nullopt);
}

ReductionOpResult
callReduction(CallType callType,
              const ::tt::target::ttnn::ReductionOpT &reductionOp,
              TensorArg input, ::ttnn::MeshDevice *device) {
  ReductionResolvedParams params = resolveReductionParams(reductionOp);

  auto makeTuple = [&](auto tag) {
    return createReductionTuple(tag, reductionOp, input, params);
  };

  auto invokeReduction = [&](auto op) {
    return callOp<ReductionOpResult>(op, callType, makeTuple, device);
  };

  switch (reductionOp.type) {
  case ::tt::target::ttnn::ReductionOpType::Sum:
    return invokeReduction(WRAP_OP(::ttnn::sum));
  case ::tt::target::ttnn::ReductionOpType::Mean:
    return invokeReduction(WRAP_OP(::ttnn::mean));
  case ::tt::target::ttnn::ReductionOpType::Max:
    return invokeReduction(WRAP_OP(::ttnn::max));
  case ::tt::target::ttnn::ReductionOpType::Min:
    return invokeReduction(WRAP_OP(::ttnn::min));
  }
  llvm_unreachable("unhandled ReductionOpType");
}

} // namespace ttnn_op_invoke
