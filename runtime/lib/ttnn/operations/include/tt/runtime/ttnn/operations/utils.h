// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_UTILS_H
#define TT_RUNTIME_TTNN_OPERATIONS_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "types_generated.h"
#include <concepts>
#include <cstdint>

namespace tt::runtime::ttnn::operations::utils {

bool isTilized(const ::tt::target::TensorRef *tensorRef);

bool inSystemMemory(const ::tt::target::TensorRef *tensorRef);

::tt::target::MemorySpace
getMemorySpace(const ::tt::target::TensorRef *tensorRef);

::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef);

::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::MemoryConfigDesc *memcfg,
                   const ::tt::target::TensorRef *tensorRef);

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::DistributionStrategy *strategy);

template <std::integral T>
inline ::ttnn::Shape toTTNNShape(const flatbuffers::Vector<T> &vec) {
  std::vector<uint32_t> rawShape;
  rawShape.reserve(vec.size());
  std::transform(
      vec.begin(), vec.end(), std::back_inserter(rawShape),
      [](const T &x) -> uint32_t { return static_cast<uint32_t>(x); });
  return ::ttnn::Shape(rawShape);
}

} // namespace tt::runtime::ttnn::operations::utils
#endif
