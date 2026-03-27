// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/sort.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include <optional>
#include <vector>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SortOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::sort(in, op->dim(), op->descending(), op->stable(),
                   outputMemoryConfig, std::nullopt);

  LOG_ASSERT(
      op->outputs()->size() == outputs.size(),
      "Number of expected outputs does not match with generated outputs.");

  if (outputs.size() >= 2) {
    ::ttnn::Tensor idxZeros = ::ttnn::zeros_like(outputs[1]);
    ::ttnn::Tensor idxSum = ::ttnn::add(idxZeros, outputs[1]);
    ::ttnn::Tensor idxTensor = idxSum.cpu();
    idxTensor = ::ttnn::to_layout(idxTensor, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
    uint16_t minVal = UINT16_MAX, maxVal = 0;
    const auto &storage = idxTensor.host_storage();
    storage.buffer().apply([&](const ::tt::tt_metal::HostBuffer &shard) {
      auto bytes = shard.view_bytes();
      auto *data = reinterpret_cast<const uint16_t *>(bytes.data());
      size_t n = bytes.size() / sizeof(uint16_t);
      for (size_t i = 0; i < n; ++i) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
      }
    });
    LOG_INFO("SORT indices: min={} max={}", (int)minVal, (int)maxVal);
  }

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::data_movement
