// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/eltwise/binary/eltwiseBinaryCompositeOp.h"
#include "operations/functions.hpp"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

EltwiseBinaryCompositeResolvedParams resolveEltwiseBinaryCompositeParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOpT,
    CallType callType) {

  EltwiseBinaryCompositeResolvedParams params;

  if (eltwiseBinaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeOpT.out),
        callType);
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
        &eltwiseBinaryCompositeScalarOpT,
    CallType callType) {

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
            *eltwiseBinaryCompositeScalarOpT.out),
        callType);

    LOG_ASSERT(operations::utils::inSystemMemory(
                   *eltwiseBinaryCompositeScalarOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEltwiseBinaryCompositeScalarTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT,
    TensorArg lhs, ::ttnn::MeshDevice *device,
    const EltwiseBinaryCompositeScalarResolvedParams &params, auto &&exponent) {
  return std::make_tuple(resolveTensorArg(lhs, tag), exponent,
                         params.outputMemoryConfig);
}

EltwiseBinaryCompositeScalarOpResult callEltwiseBinaryCompositeScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT,
    TensorArg lhs, ::ttnn::MeshDevice *device) {

  EltwiseBinaryCompositeScalarResolvedParams params =
      resolveEltwiseBinaryCompositeScalarParams(eltwiseBinaryCompositeScalarOpT,
                                                callType);

  auto makeTuple = [&](auto tag, auto exponent) {
    return createEltwiseBinaryCompositeScalarTuple(
        tag, eltwiseBinaryCompositeScalarOpT, lhs, device, params, exponent);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::visit(
        [&](auto &&exponent) {
          return std::apply(
              [&](auto &&...args) {
                return QUERY_OP_CONSTRAINTS(
                    ::ttnn::pow, device, std::forward<decltype(args)>(args)...);
              },
              makeTuple(QueryTag{}, exponent));
        },
        params.exponent);
  case CallType::QUERY_OP_RUNTIME:
    return std::visit(
        [&](auto &&exponent) {
          return std::apply(
              [&](auto &&...args) {
                return QUERY_OP_RUNTIME(::ttnn::pow, device,
                                        std::forward<decltype(args)>(args)...);
              },
              makeTuple(QueryTag{}, exponent));
        },
        params.exponent);
  case CallType::EXECUTE:
    return std::visit(
        [&](auto &&exponent) {
          return std::apply(
              [&](auto &&...args) {
                return ::ttnn::pow(std::forward<decltype(args)>(args)...);
              },
              makeTuple(ExecuteTag{}, exponent));
        },
        params.exponent);
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
