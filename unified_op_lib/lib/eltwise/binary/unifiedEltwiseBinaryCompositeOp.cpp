// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/binary/unifiedEltwiseBinaryCompositeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "utils/utils.h"
#include <operations/functions.hpp>
#include <optional>
#include <variant>

namespace unifiedOpLib {

EltwiseBinaryCompositeResolvedParams resolveEltwiseBinaryCompositeParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOpT) {

  EltwiseBinaryCompositeResolvedParams params;

  if (eltwiseBinaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeOpT.out));
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseBinaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseBinaryCompositeScalarResolvedParams
resolveEltwiseBinaryCompositeScalarParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT) {

  EltwiseBinaryCompositeScalarResolvedParams params;

  switch (eltwiseBinaryCompositeScalarOpT.rhs.type) {
  case ::tt::target::ttnn::NumberType::FP:
    params.exponent = eltwiseBinaryCompositeScalarOpT.rhs.AsFP()->value;
    break;
  case ::tt::target::ttnn::NumberType::I32:
    params.exponent = eltwiseBinaryCompositeScalarOpT.rhs.AsI32()->value;
    break;
  default:
    LOG_FATAL("unknown exponent type");
  }

  if (eltwiseBinaryCompositeScalarOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeScalarOpT.out));

    LOG_ASSERT(operations::utils::inSystemMemory(
                   *eltwiseBinaryCompositeScalarOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseBinaryCompositeScalarOpResult callEltwiseBinaryCompositeScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT,
    TensorArg lhs, ::ttnn::MeshDevice *device) {

  EltwiseBinaryCompositeScalarResolvedParams params =
      resolveEltwiseBinaryCompositeScalarParams(
          eltwiseBinaryCompositeScalarOpT);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    return std::visit(
        [&](auto &&exponent) {
          return QUERY_OP_CONSTRAINTS(::ttnn::pow, device,
                                      std::get<::ttnn::TensorSpec>(lhs),
                                      exponent, params.outputMemoryConfig);
        },
        params.exponent);
  }
  case CallType::QUERY_OP_RUNTIME: {
    return std::visit(
        [&](auto &&exponent) {
          return QUERY_OP_RUNTIME(::ttnn::pow, device,
                                  std::get<::ttnn::TensorSpec>(lhs), exponent,
                                  params.outputMemoryConfig);
        },
        params.exponent);
  }
  case CallType::EXECUTE: {
    const auto &input = *std::get<const ::ttnn::Tensor *>(lhs);

    return std::visit(
        [&](auto &&exponent) {
          return ::ttnn::pow(input, exponent, params.outputMemoryConfig);
        },
        params.exponent);
  }
  }
}

} // namespace unifiedOpLib
