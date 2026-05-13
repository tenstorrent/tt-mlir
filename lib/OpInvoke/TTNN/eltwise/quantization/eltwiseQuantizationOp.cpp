// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/eltwise/quantization/eltwiseQuantizationOp.h"
#include "operations/eltwise/quantization/quantization.hpp"
#include "operations/functions.hpp"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

EltwiseQuantizationResolvedParams resolveEltwiseQuantizationParams(
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    CallType callType) {

  EltwiseQuantizationResolvedParams params;

  if (eltwiseQuantizationOpT.axis) {
    params.axis = *(eltwiseQuantizationOpT.axis);
  }

  if (eltwiseQuantizationOpT.output_dtype.has_value()) {
    params.outputDataType = operations::utils::toTTNNDataType(
        eltwiseQuantizationOpT.output_dtype.value());
  } else if (eltwiseQuantizationOpT.out) {
    params.outputDataType =
        operations::utils::getDataType(*eltwiseQuantizationOpT.out);
  }

  if (eltwiseQuantizationOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseQuantizationOpT.out),
        callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseQuantizationOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEltwiseQuantizeDequantizeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg inputParam, TensorVariantArg<float> scaleParam,
    TensorVariantArg<int32_t> zeroPointParam, ::ttnn::MeshDevice *device,
    const EltwiseQuantizationResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(inputParam, tag),
                         resolveTensorVariantArg(scaleParam, tag),
                         resolveTensorVariantArg(zeroPointParam, tag),
                         params.axis, params.outputDataType,
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt);
}

EltwiseQuantizationOpResult callEltwiseQuantizeDequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg inputParam, TensorVariantArg<float> scaleParam,
    TensorVariantArg<int32_t> zeroPointParam, ::ttnn::MeshDevice *device) {

  EltwiseQuantizationResolvedParams params =
      resolveEltwiseQuantizationParams(eltwiseQuantizationOpT, callType);

  std::function<::ttnn::Tensor(
      const ::ttnn::Tensor &, const std::variant<::ttnn::Tensor, float> &,
      const std::variant<::ttnn::Tensor, int32_t> &, std::optional<int32_t>,
      std::optional<::ttnn::DataType>, std::optional<::ttnn::MemoryConfig>,
      std::optional<::ttnn::Tensor>)>
      func;

  if (eltwiseQuantizationOpT.type ==
      ::tt::target::ttnn::EltwiseQuantizationOpType::Quantize) {
    func = ::ttnn::quantize;
  } else if (eltwiseQuantizationOpT.type ==
             ::tt::target::ttnn::EltwiseQuantizationOpType::Dequantize) {
    func = ::ttnn::dequantize;
  } else {
    LOG_FATAL("EltwiseQuantizationOpType must be Quantize or Dequantize");
  }

  auto makeTuple = [&](auto tag) {
    return createEltwiseQuantizeDequantizeTuple(tag, eltwiseQuantizationOpT,
                                                inputParam, scaleParam,
                                                zeroPointParam, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_constraints(
              func, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::graph::query_op_runtime(
              func, device, std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return func(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

template <typename Tag>
auto createEltwiseRequantizeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg inputParam, TensorVariantArg<float> inScaleParam,
    TensorVariantArg<int32_t> inZeroPointParam,
    TensorVariantArg<float> outScaleParam,
    TensorVariantArg<int32_t> outZeroPointParam, ::ttnn::MeshDevice *device,
    const EltwiseQuantizationResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(inputParam, tag),
                         resolveTensorVariantArg(inScaleParam, tag),
                         resolveTensorVariantArg(inZeroPointParam, tag),
                         resolveTensorVariantArg(outScaleParam, tag),
                         resolveTensorVariantArg(outZeroPointParam, tag),
                         params.axis, params.outputDataType,
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt);
}

EltwiseQuantizationOpResult callEltwiseRequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg inputParam, TensorVariantArg<float> inScaleParam,
    TensorVariantArg<int32_t> inZeroPointParam,
    TensorVariantArg<float> outScaleParam,
    TensorVariantArg<int32_t> outZeroPointParam, ::ttnn::MeshDevice *device) {

  EltwiseQuantizationResolvedParams params =
      resolveEltwiseQuantizationParams(eltwiseQuantizationOpT, callType);

  LOG_ASSERT(eltwiseQuantizationOpT.type ==
                 ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize,
             "EltwiseQuantizationOpType must be Requantize");

  auto makeTuple = [&](auto tag) {
    return createEltwiseRequantizeTuple(
        tag, eltwiseQuantizationOpT, inputParam, inScaleParam, inZeroPointParam,
        outScaleParam, outZeroPointParam, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::requantize, device,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::requantize, device,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::requantize(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
