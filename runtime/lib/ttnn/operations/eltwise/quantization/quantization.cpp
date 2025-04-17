// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/quantization/quantization.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <optional>
#include <variant>

namespace tt::runtime::ttnn::operations::eltwise::quantization {
// Helper function to convert a TensorRef to a Tensor.
static ::ttnn::Tensor
convertTensorRefToTensor(const tt::target::ttnn::TensorRef *tensorRef,
                         ProgramContext &context) {
  return context.getTensorPool().getTTNNTensorAndValidate(tensorRef);
}

// Helper function to get scale value from QuantizationOpParams.
static std::variant<::ttnn::Tensor, float> getScaleValueFromQuantizationParams(
    const tt::target::ttnn::QuantizeDequantizeOpParams *params,
    ProgramContext &context) {
  if (params->scale_type() ==
      ::tt::target::ttnn::QuantizationScale::PerTensorScale) {
    return params->scale_as_PerTensorScale()->scale();
  }
  if (params->scale_type() ==
      ::tt::target::ttnn::QuantizationScale::PerAxisScale) {
    return convertTensorRefToTensor(params->scale_as_PerAxisScale()->scale(),
                                    context);
  }
  LOG_FATAL("Invalid scale type.");
  return 0.0f;
}

// Helper function to get zero point value from QuantizationOpParams.
static std::variant<::ttnn::Tensor, int32_t>
getZeroPointValueFromQuantizationParams(
    const tt::target::ttnn::QuantizeDequantizeOpParams *params,
    ProgramContext &context) {
  if (params->zero_point_type() ==
      ::tt::target::ttnn::QuantizationZeroPoint::PerTensorZeroPoint) {
    return params->zero_point_as_PerTensorZeroPoint()->zero_point();
  }
  if (params->zero_point_type() ==
      ::tt::target::ttnn::QuantizationZeroPoint::PerAxisZeroPoint) {
    return convertTensorRefToTensor(
        params->zero_point_as_PerAxisZeroPoint()->zero_point(), context);
  }
  LOG_FATAL("Invalid zero point type.");
  return 0;
}

// Helper function to handle common logic.
template <typename OpType, typename QuantizationFunc>
static void runQuantizationOp(const OpType *op, ProgramContext &context,
                              QuantizationFunc quantizationFunc) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<int32_t> axis = std::nullopt;
  if (op->axis()) {
    axis = *(op->axis());
  }

  std::optional<::ttnn::DataType> outputDataType = std::nullopt;
  if (op->output_dtype()) {
    outputDataType =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  // Call the specific quantization function.
  ::ttnn::Tensor output =
      quantizationFunc(input, axis, outputDataType, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseQuantizationOp *op,
         ProgramContext &context) {
  using namespace ::tt::target::ttnn;

  switch (op->type()) {
  case EltwiseQuantizationOpType::Quantize: {
    const auto *const params = op->params_as_QuantizeDequantizeOpParams();
    auto scale = getScaleValueFromQuantizationParams(params, context);
    auto zeroPoint = getZeroPointValueFromQuantizationParams(params, context);
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::quantize(input, scale, zeroPoint, axis, outputDataType,
                                  memoryConfig,
                                  /* optional_output_tensor=*/std::nullopt);
        });
    break;
  }
  case EltwiseQuantizationOpType::Dequantize: {
    const auto *const params = op->params_as_QuantizeDequantizeOpParams();
    auto scale = getScaleValueFromQuantizationParams(params, context);
    auto zeroPoint = getZeroPointValueFromQuantizationParams(params, context);
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::dequantize(input, scale, zeroPoint, axis,
                                    outputDataType, memoryConfig,
                                    /* optional_output_tensor=*/std::nullopt);
        });
    break;
  }
  case EltwiseQuantizationOpType::Requantize: {
    const auto *const params = op->params_as_RequantizeOpParams();
    auto inScale = [&]() -> std::variant<::ttnn::Tensor, float> {
      if (params->in_scale_type() == QuantizationScale::PerTensorScale) {
        return params->in_scale_as_PerTensorScale()->scale();
      }
      if (params->in_scale_type() == QuantizationScale::PerAxisScale) {
        return convertTensorRefToTensor(
            params->in_scale_as_PerAxisScale()->scale(), context);
      }
      LOG_FATAL("Invalid input scale type.");
      return 0.0f;
    }();
    auto outScale = [&]() -> std::variant<::ttnn::Tensor, float> {
      if (params->out_scale_type() == QuantizationScale::PerTensorScale) {
        return params->out_scale_as_PerTensorScale()->scale();
      }
      if (params->out_scale_type() == QuantizationScale::PerAxisScale) {
        return convertTensorRefToTensor(
            params->out_scale_as_PerAxisScale()->scale(), context);
      }
      LOG_FATAL("Invalid output scale type.");
      return 0.0f;
    }();
    auto inZeroPoint = [&]() -> std::variant<::ttnn::Tensor, int32_t> {
      if (params->in_zero_point_type() ==
          QuantizationZeroPoint::PerTensorZeroPoint) {
        return params->in_zero_point_as_PerTensorZeroPoint()->zero_point();
      }
      if (params->in_zero_point_type() ==
          QuantizationZeroPoint::PerAxisZeroPoint) {
        return convertTensorRefToTensor(
            params->in_zero_point_as_PerAxisZeroPoint()->zero_point(), context);
      }
      LOG_FATAL("Invalid input zero point type.");
      return 0;
    }();
    auto outZeroPoint = [&]() -> std::variant<::ttnn::Tensor, int32_t> {
      if (params->out_zero_point_type() ==
          QuantizationZeroPoint::PerTensorZeroPoint) {
        return params->out_zero_point_as_PerTensorZeroPoint()->zero_point();
      }
      if (params->out_zero_point_type() ==
          QuantizationZeroPoint::PerAxisZeroPoint) {
        return convertTensorRefToTensor(
            params->out_zero_point_as_PerAxisZeroPoint()->zero_point(),
            context);
      }
      LOG_FATAL("Invalid output zero point type.");
      return 0;
    }();
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::requantize(input, inScale, inZeroPoint, outScale,
                                    outZeroPoint, axis, outputDataType,
                                    memoryConfig,
                                    /* optional_output_tensor=*/std::nullopt);
        });
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::quantization
