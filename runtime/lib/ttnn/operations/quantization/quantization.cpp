// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/quantization/quantization.h"

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
  const ::ttnn::Tensor &input = tensorPool.getAndValidate(op->input());

  std::optional<int32_t> axis = std::optional<int32_t>();
  std::optional<::ttnn::DataType> outputDataType =
      std::optional<::ttnn::DataType>();
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  if (op->axis()) {
    axis = *(op->axis());
  }

  if (op->output_dtype()) {
    outputDataType =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  // Call the specific quantization function.
  ::ttnn::Tensor output =
      quantizationFunc(input, axis, outputDataType, memoryConfig);

  tensorPool.insertAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::QuantizeOp *op, ProgramContext &context) {
  runQuantizationOp(
      op, context,
      [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
          std::optional<::ttnn::DataType> outputDataType,
          std::optional<::ttnn::MemoryConfig> memoryConfig) {
        return ::ttnn::quantize(input, op->scale(), op->zero_point(), axis,
                                outputDataType, memoryConfig,
                                /* optional_output_tensor=*/std::nullopt);
      });
}

void run(const ::tt::target::ttnn::DequantizeOp *op, ProgramContext &context) {
  runQuantizationOp(
      op, context,
      [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
          std::optional<::ttnn::DataType> outputDataType,
          std::optional<::ttnn::MemoryConfig> memoryConfig) {
        return ::ttnn::dequantize(input, op->scale(), op->zero_point(), axis,
                                  outputDataType, memoryConfig,
                                  /* optional_output_tensor=*/std::nullopt);
      });
}

void run(const ::tt::target::ttnn::RequantizeOp *op, ProgramContext &context) {
  runQuantizationOp(
      op, context,
      [&](const ::ttnn::Tensor &input, std::optional<int32_t> axis,
          std::optional<::ttnn::DataType> outputDataType,
          std::optional<::ttnn::MemoryConfig> memoryConfig) {
        return ::ttnn::requantize(input, op->in_scale(), op->in_zero_point(),
                                  op->out_scale(), op->out_zero_point(), axis,
                                  outputDataType, memoryConfig,
                                  /* optional_output_tensor=*/std::nullopt);
      });
}

} // namespace tt::runtime::ttnn::operations::quantization
