// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/quantization/unifiedEltwiseQuantizationOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "utils/utils.h"
#include <operations/eltwise/quantization/quantization.hpp>
#include <operations/functions.hpp>
#include <optional>
#include <ttnn/graph/graph_query_op_runtime.hpp>
#include <variant>

namespace unifiedOpLib {

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
    LOG_ASSERT(false &&
               "EltwiseQuantizationOpType must be Quantize or Dequantize");
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return ::ttnn::graph::query_op_constraints(
        func, device, std::get<::ttnn::TensorSpec>(inputParam),
        std::get<::ttnn::TensorSpec>(scaleParam),
        std::get<::ttnn::TensorSpec>(zeroPointParam), params.axis,
        params.outputDataType, params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt);
  case CallType::QUERY_OP_RUNTIME:
    return ::ttnn::graph::query_op_runtime(
        func, device, std::get<::ttnn::TensorSpec>(inputParam),
        std::get<::ttnn::TensorSpec>(scaleParam),
        std::get<::ttnn::TensorSpec>(zeroPointParam), params.axis,
        params.outputDataType, params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt);
  case CallType::EXECUTE: {
    const auto &input = *std::get<const ::ttnn::Tensor *>(inputParam);
    const auto &scale =
        std::get<std::variant<::ttnn::Tensor, float>>(scaleParam);
    const auto &zeroPoint =
        std::get<std::variant<::ttnn::Tensor, int32_t>>(zeroPointParam);

    return func(input, scale, zeroPoint, params.axis, params.outputDataType,
                params.outputMemoryConfig,
                /*optional_output_tensor=*/std::nullopt);
  }
  }
}

EltwiseQuantizationOpResult callEltwiseRequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg input_param, TensorVariantArg<float> in_scale_param,
    TensorVariantArg<int32_t> in_zero_point_param,
    TensorVariantArg<float> out_scale_param,
    TensorVariantArg<int32_t> out_zero_point_param,
    ::ttnn::MeshDevice *device) {

  EltwiseQuantizationResolvedParams params =
      resolveEltwiseQuantizationParams(eltwiseQuantizationOpT, callType);

  LOG_ASSERT(eltwiseQuantizationOpT.type ==
                 ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize &&
             "EltwiseQuantizationOpType must be Requantize");

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::requantize, device, std::get<::ttnn::TensorSpec>(input_param),
        std::get<::ttnn::TensorSpec>(in_scale_param),
        std::get<::ttnn::TensorSpec>(in_zero_point_param),
        std::get<::ttnn::TensorSpec>(out_scale_param),
        std::get<::ttnn::TensorSpec>(out_zero_point_param), params.axis,
        params.outputDataType, params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(
        ::ttnn::requantize, device, std::get<::ttnn::TensorSpec>(input_param),
        std::get<::ttnn::TensorSpec>(in_scale_param),
        std::get<::ttnn::TensorSpec>(in_zero_point_param),
        std::get<::ttnn::TensorSpec>(out_scale_param),
        std::get<::ttnn::TensorSpec>(out_zero_point_param), params.axis,
        params.outputDataType, params.outputMemoryConfig,
        /*optional_output_tensor=*/std::nullopt);
  case CallType::EXECUTE: {
    const auto &input = *std::get<const ::ttnn::Tensor *>(input_param);
    const auto &in_scale =
        std::get<std::variant<::ttnn::Tensor, float>>(in_scale_param);
    const auto &in_zero_point =
        std::get<std::variant<::ttnn::Tensor, int32_t>>(in_zero_point_param);
    const auto &out_scale =
        std::get<std::variant<::ttnn::Tensor, float>>(out_scale_param);
    const auto &out_zero_point =
        std::get<std::variant<::ttnn::Tensor, int32_t>>(out_zero_point_param);

    return ::ttnn::requantize(input, in_scale, in_zero_point, out_scale,
                              out_zero_point, params.axis,
                              params.outputDataType, params.outputMemoryConfig,
                              /*optional_output_tensor=*/std::nullopt);
  }
  }
}

} // namespace unifiedOpLib
