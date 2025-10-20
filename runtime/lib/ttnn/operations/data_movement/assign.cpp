// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/assign.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::AssignOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::MemoryConfig> memoryConfigOpt =
      op->output_memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(
                    op->output()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->output_memory_config());

  ::ttnn::MemoryConfig memoryConfig =
      memoryConfigOpt.value_or(::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});

  std::optional<::ttnn::DataType> outputDtype = std::nullopt;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*op->output_dtype());
  }

  std::optional<::ttnn::Tensor> optionalOutputTensor = std::nullopt;
  ::ttnn::Tensor output =
      ::ttnn::assign(in, memoryConfig, outputDtype, optionalOutputTensor);
  tensorPool.insertTTNNTensorAndValidate(op->output(), output);
}
} // namespace tt::runtime::ttnn::operations::data_movement
