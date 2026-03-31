// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/worker/tensor_data_cache.h"
#include <cstdint>
#include <numeric>
#include <tt_stl/reflection.hpp>

namespace tt::runtime::distributed::worker {

std::shared_ptr<std::vector<uint8_t>> TensorDataCache::insert_or_assign(
    const uint8_t *tensorDataPtr, std::vector<uint32_t> shape,
    std::vector<uint32_t> stride, uint32_t itemSize,
    ::tt::target::DataType dataType) {
  uint64_t tensorHash =
      GetTensorHash(tensorDataPtr, shape, stride, itemSize, dataType);
  if (_pool.contains(tensorHash)) {
    if (auto cached = _pool.at(tensorHash).lock()) {
      return cached;
    }
  }

  uint64_t numElements =
      std::accumulate(shape.begin(), shape.end(), static_cast<std::uint64_t>(1),
                      std::multiplies<std::uint64_t>());
  size_t length = numElements * itemSize;

  auto newVector = std::make_shared<std::vector<uint8_t>>(
      tensorDataPtr, tensorDataPtr + length);
  _pool[tensorHash] = newVector;

  return newVector;
}

uint64_t TensorDataCache::GetTensorHash(const uint8_t *tensorDataPtr,
                                        std::vector<uint32_t> shape,
                                        std::vector<uint32_t> stride,
                                        uint32_t itemSize,
                                        ::tt::target::DataType dataType) {
  return ttsl::hash::hash_objects_with_default_seed(
      reinterpret_cast<uintptr_t>(tensorDataPtr), shape, stride, itemSize,
      static_cast<uint16_t>(dataType));
}

} // namespace tt::runtime::distributed::worker
