// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/quantization/quantization.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <optional>

namespace tt::runtime::ttnn::operations::eltwise::quantization {

// Helper function to handle common logic.
template <typename OpType, typename QuantizationFunc>
void runQuantizationOp(const OpType *op, ProgramContext &context,
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
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseQuantizationOpType::Quantize:
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::quantize(
              input, op->params_as_QuantizeDequantizeOpParams()->scale(),
              op->params_as_QuantizeDequantizeOpParams()->zero_point(), axis,
              outputDataType, memoryConfig,
              /* optional_output_tensor=*/std::nullopt);
        });
    break;
  case ::tt::target::ttnn::EltwiseQuantizationOpType::Dequantize:
    runQuantizationOp(
        op, context,
        [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
            std::optional<::ttnn::DataType> outputDataType,
            std::optional<::ttnn::MemoryConfig> memoryConfig) {
          return ::ttnn::dequantize(
              input, op->params_as_QuantizeDequantizeOpParams()->scale(),
              op->params_as_QuantizeDequantizeOpParams()->zero_point(), axis,
              outputDataType, memoryConfig,
              /* optional_output_tensor=*/std::nullopt);
        });
    break;
  case ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize:
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
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::quantization
