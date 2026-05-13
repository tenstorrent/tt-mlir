// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Quantization/EltwiseQuantizationOp.h"
#include "operations/eltwise/quantization/quantization.hpp"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/functions.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

EltwiseQuantizationResolvedParams resolveEltwiseQuantizationParams(
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp) {

  EltwiseQuantizationResolvedParams params;

  if (eltwiseQuantizationOp.axis) {
    params.axis = *(eltwiseQuantizationOp.axis);
  }

  if (eltwiseQuantizationOp.output_dtype.has_value()) {
    params.outputDataType = operations::utils::toTTNNDataType(
        eltwiseQuantizationOp.output_dtype.value());
  } else if (eltwiseQuantizationOp.out) {
    params.outputDataType =
        operations::utils::getDataType(*eltwiseQuantizationOp.out);
  }

  if (eltwiseQuantizationOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseQuantizationOp.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseQuantizationOp.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEltwiseQuantizeDequantizeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp,
    TensorArg inputParam, TensorVariantArg<float> scaleParam,
    TensorVariantArg<int32_t> zeroPointParam,
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
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp,
    TensorArg inputParam, TensorVariantArg<float> scaleParam,
    TensorVariantArg<int32_t> zeroPointParam, ::ttnn::MeshDevice *device) {

  EltwiseQuantizationResolvedParams params =
      resolveEltwiseQuantizationParams(eltwiseQuantizationOp);

  std::function<::ttnn::Tensor(
      const ::ttnn::Tensor &, const std::variant<::ttnn::Tensor, float> &,
      const std::variant<::ttnn::Tensor, int32_t> &, std::optional<int32_t>,
      std::optional<::ttnn::DataType>, std::optional<::ttnn::MemoryConfig>,
      std::optional<::ttnn::Tensor>)>
      func;

  if (eltwiseQuantizationOp.type ==
      ::tt::target::ttnn::EltwiseQuantizationOpType::Quantize) {
    func = ::ttnn::quantize;
  } else if (eltwiseQuantizationOp.type ==
             ::tt::target::ttnn::EltwiseQuantizationOpType::Dequantize) {
    func = ::ttnn::dequantize;
  } else {
    LOG_FATAL("EltwiseQuantizationOpType must be Quantize or Dequantize");
  }

  auto makeTuple = [&](auto tag) {
    return createEltwiseQuantizeDequantizeTuple(tag, eltwiseQuantizationOp,
                                                inputParam, scaleParam,
                                                zeroPointParam, params);
  };

  return callOp<EltwiseQuantizationOpResult>(func, callType, makeTuple, device);
}

template <typename Tag>
auto createEltwiseRequantizeTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp,
    TensorArg inputParam, TensorVariantArg<float> inScaleParam,
    TensorVariantArg<int32_t> inZeroPointParam,
    TensorVariantArg<float> outScaleParam,
    TensorVariantArg<int32_t> outZeroPointParam,
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
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp,
    TensorArg inputParam, TensorVariantArg<float> inScaleParam,
    TensorVariantArg<int32_t> inZeroPointParam,
    TensorVariantArg<float> outScaleParam,
    TensorVariantArg<int32_t> outZeroPointParam, ::ttnn::MeshDevice *device) {

  EltwiseQuantizationResolvedParams params =
      resolveEltwiseQuantizationParams(eltwiseQuantizationOp);

  LOG_ASSERT(eltwiseQuantizationOp.type ==
                 ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize,
             "EltwiseQuantizationOpType must be Requantize");

  auto makeTuple = [&](auto tag) {
    return createEltwiseRequantizeTuple(
        tag, eltwiseQuantizationOp, inputParam, inScaleParam, inZeroPointParam,
        outScaleParam, outZeroPointParam, params);
  };

  return callOp<EltwiseQuantizationOpResult>(WRAP_OP(::ttnn::requantize),
                                             callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
