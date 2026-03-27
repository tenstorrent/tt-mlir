// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/typecast.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"
#include "ttnn/operations/core/core.hpp"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::TypecastOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttnn::DataType targetDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  ::ttnn::Tensor out =
      ::ttnn::typecast(inputTensor, targetDataType, memoryConfig);

  auto outShape = out.logical_shape();
  if (targetDataType == ::ttnn::DataType::INT32 &&
      outShape.size() == 2 && outShape[0] == 128 && outShape[1] == 32) {
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
    LOG_INFO("TYPECAST [128x32] si32 indices: min={} max={}", (int)minVal, (int)maxVal);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
