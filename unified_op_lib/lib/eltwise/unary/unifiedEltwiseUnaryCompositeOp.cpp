// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/unary/unifiedEltwiseUnaryCompositeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "utils/utils.h"
#include <operations/functions.hpp>
#include <optional>
#include <variant>

namespace unifiedOpLib {

EltwiseUnaryCompositeResolvedParams resolveEltwiseUnaryCompositeParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT) {

  EltwiseUnaryCompositeResolvedParams params;

  params.fastApproxMode = false;

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out));
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
        &eltwiseUnaryCompositeOpT) {

  EltwiseUnaryCompositeClampScalarResolvedParams params;

  const auto *clampParams =
      eltwiseUnaryCompositeOpT.params.AsClampScalarOpParams();

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
            *eltwiseUnaryCompositeOpT.out));
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
        &eltwiseUnaryCompositeOpT) {

  EltwiseUnaryCompositeClampTensorResolvedParams params;

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out));
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig) {

  EltwiseUnaryCompositeClampScalarResolvedParams params =
      resolveEltwiseUnaryCompositeClampScalarParams(eltwiseUnaryCompositeOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    return std::visit(
        [&](auto &&min, auto &&max) {
          return QUERY_OP_CONSTRAINTS(::ttnn::clamp, device,
                                      std::get<::ttnn::TensorSpec>(input), min,
                                      max, params.outputMemoryConfig);
        },
        params.min, params.max);
  }
  case CallType::QUERY_OP_RUNTIME: {
    return std::visit(
        [&](auto &&min, auto &&max) {
          return QUERY_OP_RUNTIME(::ttnn::clamp, device,
                                  std::get<::ttnn::TensorSpec>(input), min, max,
                                  params.outputMemoryConfig);
        },
        params.min, params.max);
  }
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);

    return std::visit(
        [&](auto &&min, auto &&max) {
          return ::ttnn::clamp(in, min, max, params.outputMemoryConfig);
        },
        params.min, params.max);
  }
  }
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max, ::ttnn::MeshDevice *device,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig) {

  EltwiseUnaryCompositeClampTensorResolvedParams params =
      resolveEltwiseUnaryCompositeClampTensorParams(eltwiseUnaryCompositeOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::clamp, device, std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(min), std::get<::ttnn::TensorSpec>(max),
        params.outputMemoryConfig);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(
        ::ttnn::clamp, device, std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(min), std::get<::ttnn::TensorSpec>(max),
        params.outputMemoryConfig);
  case CallType::EXECUTE: {
    const auto &inTensor = *std::get<const ::ttnn::Tensor *>(input);
    const auto &minTensor = *std::get<const ::ttnn::Tensor *>(min);
    const auto &maxTensor = *std::get<const ::ttnn::Tensor *>(max);

    return ::ttnn::clamp(inTensor, minTensor, maxTensor,
                         params.outputMemoryConfig);
  }
  }
}

} // namespace unifiedOpLib
