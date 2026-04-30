// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/gather.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {

namespace {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
bool allNonNegative(const ::ttnn::Tensor &index) {
  std::vector<int32_t> values = index.to_vector<int32_t>();
  return std::all_of(values.begin(), values.end(),
                     [](int32_t v) { return v >= 0; });
}
#endif
} // namespace

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

  ::ttnn::Tensor newIndex = index;
  if (index.dtype() == ::ttnn::DataType::INT32) {
    DEBUG_ASSERT(allNonNegative(index),
                 "ttnn::gather INT32 index must be non-negative");
    newIndex = ::ttnn::typecast(index, ::ttnn::DataType::UINT32);
  }

  ::ttnn::Tensor out =
      ::ttnn::gather(input, dim, newIndex, /*sparse_grad=*/false,
                     outputMemoryConfig, std::nullopt, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
