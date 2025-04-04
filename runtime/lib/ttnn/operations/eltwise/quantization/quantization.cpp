// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/quantization/quantization.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <optional>

namespace tt::runtime::ttnn::operations::quantization {

// Helper function to handle common logic.
template <typename OpType, typename QuantizationFunc>
void runQuantizationOp(const OpType *op, ProgramContext &context,
                       QuantizationFunc quantizationFunc) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getAndValidate(op->ins()->Get(0));

  std::optional<int32_t> axis = std::optional<int32_t>();
  std::optional<::ttnn::DataType> outputDataType =
      std::optional<::ttnn::DataType>();
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 memoryConfig.has_value(),
             "Memory config must exist for device tensors");

  // Handle parameters based on operation type
  if (op->params_type() ==
      ::tt::target::ttnn::EltwiseOpParams::QuantizationOpParams) {
    auto params = op->params_as_QuantizationOpParams();
    if (params->axis()) {
      axis = *(params->axis());
    }
    if (params->output_dtype()) {
      outputDataType =
          ::tt::runtime::ttnn::utils::toTTNNDataType(*(params->output_dtype()));
    }
  } else if (op->params_type() ==
             ::tt::target::ttnn::EltwiseOpParams::RequantizeOpParams) {
    auto params = op->params_as_RequantizeOpParams();
    if (params->axis()) {
      axis = *(params->axis());
    }
    if (params->output_dtype()) {
      outputDataType =
          ::tt::runtime::ttnn::utils::toTTNNDataType(*(params->output_dtype()));
    }
  }

  // Call the specific quantization function.
  ::ttnn::Tensor output =
      quantizationFunc(input, axis, outputDataType, memoryConfig);

  tensorPool.insertAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Quantize:
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::quantize(
              input, op->params_as_QuantizationOpParams()->scale(),
              op->params_as_QuantizationOpParams()->zero_point(), axis,
              outputDataType, memoryConfig,
              /* optional_output_tensor=*/std::nullopt);
        });
    break;
  case ::tt::target::ttnn::EltwiseOpType::Dequantize:
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::dequantize(
              input, op->params_as_QuantizationOpParams()->scale(),
              op->params_as_QuantizationOpParams()->zero_point(), axis,
              outputDataType, memoryConfig,
              /* optional_output_tensor=*/std::nullopt);
        });
    break;
  case ::tt::target::ttnn::EltwiseOpType::Requantize:
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::requantize(
              input, op->params_as_RequantizeOpParams()->in_scale(),
              op->params_as_RequantizeOpParams()->in_zero_point(),
              op->params_as_RequantizeOpParams()->out_scale(),
              op->params_as_RequantizeOpParams()->out_zero_point(), axis,
              outputDataType, memoryConfig,
              /* optional_output_tensor=*/std::nullopt);
        });
    break;
  default:
    LOG_FATAL("Unsupported Quantization operation");
  }
}

} // namespace tt::runtime::ttnn::operations::quantization
