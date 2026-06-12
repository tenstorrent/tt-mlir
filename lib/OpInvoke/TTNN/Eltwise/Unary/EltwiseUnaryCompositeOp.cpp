// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Unary/EltwiseUnaryCompositeOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

#include <optional>
#include <tuple>

namespace ttnn_op_invoke {

EltwiseUnaryCompositeResolvedParams resolveEltwiseUnaryCompositeParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOp) {

  EltwiseUnaryCompositeResolvedParams params;

  params.fastApproxMode = false;

  if (eltwiseUnaryCompositeOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOp.out));
    TT_INVOKE_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOp.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeClampScalarResolvedParams
resolveEltwiseUnaryCompositeClampScalarParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOp) {

  EltwiseUnaryCompositeClampScalarResolvedParams params;

  const ::tt::target::ttnn::ClampScalarOpParamsT *clampParams =
      eltwiseUnaryCompositeOp.params.AsClampScalarOpParams();

  TT_INVOKE_ASSERT(clampParams->min.type == clampParams->max.type,
                   "Clamp scalar min/max types must match");

  switch (clampParams->min.type) {
  case ::tt::target::ttnn::NumberType::FP:
    params.min = clampParams->min.AsFP()->value;
    params.max = clampParams->max.AsFP()->value;
    break;
  case ::tt::target::ttnn::NumberType::I32:
    params.min = clampParams->min.AsI32()->value;
    params.max = clampParams->max.AsI32()->value;
    break;
  default:
    llvm::report_fatal_error("unknown clamp scalar min/max type");
  }

  if (eltwiseUnaryCompositeOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOp.out));
    TT_INVOKE_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOp.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeClampTensorResolvedParams
resolveEltwiseUnaryCompositeClampTensorParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOp) {

  EltwiseUnaryCompositeClampTensorResolvedParams params;

  if (eltwiseUnaryCompositeOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOp.out));
    TT_INVOKE_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOp.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEltwiseUnaryCompositeClampScalarTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT &eltwiseUnaryCompositeOp,
    TensorArg input,
    const EltwiseUnaryCompositeClampScalarResolvedParams &params) {
  TT_INVOKE_ASSERT(params.min.has_value() && params.max.has_value(),
                   "clamp scalar min/max not resolved");
  return std::make_tuple(resolveTensorArg(input, tag), *params.min, *params.max,
                         params.outputMemoryConfig);
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT &eltwiseUnaryCompositeOp,
    TensorArg input, ::ttnn::MeshDevice *device) {

  EltwiseUnaryCompositeClampScalarResolvedParams params =
      resolveEltwiseUnaryCompositeClampScalarParams(eltwiseUnaryCompositeOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeClampScalarTuple(
        tag, eltwiseUnaryCompositeOp, input, params);
  };

  return callOp<EltwiseUnaryCompositeOpResult>(WRAP_OP(::ttnn::clamp), callType,
                                               makeTuple, device);
}

template <typename Tag>
auto createEltwiseUnaryCompositeClampTensorTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT &eltwiseUnaryCompositeOp,
    TensorArg input, TensorArg min, TensorArg max,
    const EltwiseUnaryCompositeClampTensorResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         resolveTensorArg(min, tag), resolveTensorArg(max, tag),
                         params.outputMemoryConfig);
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT &eltwiseUnaryCompositeOp,
    TensorArg input, TensorArg min, TensorArg max, ::ttnn::MeshDevice *device) {

  EltwiseUnaryCompositeClampTensorResolvedParams params =
      resolveEltwiseUnaryCompositeClampTensorParams(eltwiseUnaryCompositeOp);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeClampTensorTuple(
        tag, eltwiseUnaryCompositeOp, input, min, max, params);
  };

  return callOp<EltwiseUnaryCompositeOpResult>(WRAP_OP(::ttnn::clamp), callType,
                                               makeTuple, device);
}

} // namespace ttnn_op_invoke
