// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/concat.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ConcatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(input);
    inputs.push_back(in);
  }
  int32_t dim = op->dim();
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  ::ttnn::Tensor out = ::ttnn::concat(inputs, dim, memoryConfig);

  auto outShape = out.logical_shape();
  if (out.dtype() == ::ttnn::DataType::INT32 &&
      outShape.size() == 2 && outShape[0] == 512 && outShape[1] == 2) {
    ::ttnn::Tensor outZeros = ::ttnn::zeros_like(out);
    ::ttnn::Tensor outSum = ::ttnn::add(outZeros, out);
    ::ttnn::Tensor hostOut = outSum.cpu();
    hostOut = ::ttnn::to_layout(hostOut, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
    int32_t minVal = INT32_MAX, maxVal = INT32_MIN;
    const auto &storage = hostOut.host_storage();
    storage.buffer().apply([&](const ::tt::tt_metal::HostBuffer &shard) {
      auto bytes = shard.view_bytes();
      auto *data = reinterpret_cast<const int32_t *>(bytes.data());
      size_t n = bytes.size() / sizeof(int32_t);
      for (size_t i = 0; i < n; ++i) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
      }
    });
    LOG_INFO("CONCAT [512x2] si32 indices: min={} max={}", (int)minVal, (int)maxVal);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
