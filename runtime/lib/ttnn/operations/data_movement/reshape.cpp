// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  ::ttnn::Tensor out = ::ttnn::reshape(in, shape, memoryConfig);

  auto outShape = out.logical_shape();
  if (out.dtype() == ::ttnn::DataType::INT32 &&
      outShape.size() == 2 && outShape[0] == 512 && outShape[1] == 1) {
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
    LOG_INFO("RESHAPE [512x1] si32 indices: min={} max={}", (int)minVal, (int)maxVal);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
