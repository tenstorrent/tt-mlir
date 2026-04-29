// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/gather.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::GatherOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &index =
      tensorPool.getTTNNTensorAndValidate(op->index());
  int32_t dim = op->dim();

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  // Indices must be non negative, otherwise this will break
  ::ttnn::Tensor newIndex = index;
  if (index.dtype() == ::tt::tt_metal::DataType::INT32) {
    newIndex = ::ttnn::typecast(index, ::tt::tt_metal::DataType::UINT32);
  }

  ::ttnn::Tensor out =
      ::ttnn::gather(input, dim, newIndex, /*sparse_grad=*/false,
                     outputMemoryConfig, std::nullopt, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
