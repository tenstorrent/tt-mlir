// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/eltwise/unary/eltwiseUnaryCompositeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <tuple>
#include <variant>

namespace ttnn_op_invoke {

EltwiseUnaryCompositeResolvedParams resolveEltwiseUnaryCompositeParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    CallType callType) {

  EltwiseUnaryCompositeResolvedParams params;

  params.fastApproxMode = false;

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out),
        callType);
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeClampScalarResolvedParams
resolveEltwiseUnaryCompositeClampScalarParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    CallType callType) {

  EltwiseUnaryCompositeClampScalarResolvedParams params;

  const auto *clampParams =
      eltwiseUnaryCompositeOpT.params.AsClampScalarOpParams();

  LOG_ASSERT(clampParams->min.type == clampParams->max.type,
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
    LOG_FATAL("unknown clamp scalar min/max type");
  }

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out),
        callType);
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeClampTensorResolvedParams
resolveEltwiseUnaryCompositeClampTensorParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    CallType callType) {

  EltwiseUnaryCompositeClampTensorResolvedParams params;

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out),
        callType);
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEltwiseUnaryCompositeClampScalarTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device,
    const EltwiseUnaryCompositeClampScalarResolvedParams &params) {
  LOG_ASSERT(params.min.has_value() && params.max.has_value(),
             "clamp scalar min/max not resolved");
  return std::make_tuple(resolveTensorArg(input, tag), *params.min, *params.max,
                         params.outputMemoryConfig);
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device) {

  EltwiseUnaryCompositeClampScalarResolvedParams params =
      resolveEltwiseUnaryCompositeClampScalarParams(eltwiseUnaryCompositeOpT,
                                                    callType);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeClampScalarTuple(
        tag, eltwiseUnaryCompositeOpT, input, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::clamp, device,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::clamp, device,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));

  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::clamp(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

template <typename Tag>
auto createEltwiseUnaryCompositeClampTensorTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max, ::ttnn::MeshDevice *device,
    const EltwiseUnaryCompositeClampTensorResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         resolveTensorArg(min, tag), resolveTensorArg(max, tag),
                         params.outputMemoryConfig);
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max, ::ttnn::MeshDevice *device) {

  EltwiseUnaryCompositeClampTensorResolvedParams params =
      resolveEltwiseUnaryCompositeClampTensorParams(eltwiseUnaryCompositeOpT,
                                                    callType);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeClampTensorTuple(
        tag, eltwiseUnaryCompositeOpT, input, min, max, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::clamp, device,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::clamp, device,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::clamp(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
