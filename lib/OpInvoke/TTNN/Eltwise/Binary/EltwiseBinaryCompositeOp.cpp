// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryCompositeOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>

namespace ttnn_op_invoke {

EltwiseBinaryCompositeResolvedParams resolveEltwiseBinaryCompositeParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOp) {

  EltwiseBinaryCompositeResolvedParams params;

  if (eltwiseBinaryCompositeOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeOp.out));
    TT_INVOKE_ASSERT(
        operations::utils::inSystemMemory(*eltwiseBinaryCompositeOp.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseBinaryCompositeScalarResolvedParams
resolveEltwiseBinaryCompositeScalarParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOp) {

  EltwiseBinaryCompositeScalarResolvedParams params;

  TT_INVOKE_ASSERT(eltwiseBinaryCompositeScalarOp.rhs.type ==
                           ::tt::target::ttnn::NumberType::FP ||
                       eltwiseBinaryCompositeScalarOp.rhs.type ==
                           ::tt::target::ttnn::NumberType::I32,
                   "Exponent must be either FP or I32");

  if (eltwiseBinaryCompositeScalarOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeScalarOp.out));

    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(
                         *eltwiseBinaryCompositeScalarOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag, typename Scalar>
auto createEltwiseBinaryCompositeScalarTuple(
    Tag tag, TensorArg lhs, Scalar rhs,
    const EltwiseBinaryCompositeScalarResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(lhs, tag), rhs,
                         params.outputMemoryConfig);
}

EltwiseBinaryCompositeScalarOpResult callEltwiseBinaryCompositeScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOp,
    TensorArg lhs, ::ttnn::MeshDevice *device) {

  EltwiseBinaryCompositeScalarResolvedParams params =
      resolveEltwiseBinaryCompositeScalarParams(eltwiseBinaryCompositeScalarOp);

  auto invokeWithScalar =
      [&](auto scalar) -> EltwiseBinaryCompositeScalarOpResult {
    auto makeTuple = [&](auto tag) {
      return createEltwiseBinaryCompositeScalarTuple(tag, lhs, scalar, params);
    };

    return callOp<EltwiseBinaryCompositeScalarOpResult>(
        WRAP_OP(::ttnn::pow), callType, makeTuple, device);
  };

  const auto &rhs = eltwiseBinaryCompositeScalarOp.rhs;
  if (rhs.type == ::tt::target::ttnn::NumberType::FP) {
    return invokeWithScalar(rhs.AsFP()->value);
  }
  return invokeWithScalar(rhs.AsI32()->value);
}

} // namespace ttnn_op_invoke
