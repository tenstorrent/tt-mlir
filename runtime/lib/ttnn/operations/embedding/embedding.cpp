// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/embedding/embedding.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::embedding {
void run(const ::tt::target::ttnn::EmbeddingOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  // default params for embedding op
  std::optional<int> padToken = std::nullopt;
  ::ttnn::Layout layout = utils::isTilized(op->out())
                              ? ::ttnn::TILE_LAYOUT
                              : ::ttnn::ROW_MAJOR_LAYOUT;
  auto embeddingsType = ::ttnn::prim::EmbeddingsType::GENERIC;
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  {
    auto inShape = input.logical_shape();
    if (input.dtype() == ::ttnn::DataType::UINT32 &&
        inShape.size() == 2 && inShape[0] == 1 && inShape[1] == 512) {
      ::ttnn::Tensor inZeros = ::ttnn::zeros_like(input);
      ::ttnn::Tensor inSum = ::ttnn::add(inZeros, input);
      ::ttnn::Tensor hostIn = inSum.cpu();
      hostIn = ::ttnn::to_layout(hostIn, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
      uint32_t minVal = UINT32_MAX, maxVal = 0;
      const auto &storage = hostIn.host_storage();
      storage.buffer().apply([&](const ::tt::tt_metal::HostBuffer &shard) {
        auto bytes = shard.view_bytes();
        auto *data = reinterpret_cast<const uint32_t *>(bytes.data());
        size_t n = bytes.size() / sizeof(uint32_t);
        for (size_t i = 0; i < n; ++i) {
          if (data[i] < minVal) minVal = data[i];
          if (data[i] > maxVal) maxVal = data[i];
        }
      });
      LOG_INFO("EMBEDDING [1x512] ui32 indices: min={} max={}", (unsigned)minVal, (unsigned)maxVal);
    }
  }

  ::ttnn::Tensor out =
      ::ttnn::embedding(input, weight, padToken, layout, embeddingsType,
                        outputDataType, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::embedding
