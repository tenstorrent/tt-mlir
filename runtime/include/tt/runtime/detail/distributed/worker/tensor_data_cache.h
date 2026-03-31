// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_TENSOR_DATA_CACHE_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_TENSOR_DATA_CACHE_H

#include "ttmlir/Target/Common/types_generated.h"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tt::runtime::distributed::worker {

// Cache for immutable tensor data to enable replicated data sharing across
// workers. Returns a shared_ptr<void> that points to the raw data buffer and
// shares ownership with the underlying storage, so the buffer stays alive as
// long as any consumer holds the returned shared_ptr.
class TensorDataCache {
public:
  TensorDataCache() = default;
  ~TensorDataCache() = default;

  // Returns a shared_ptr<void> pointing to the cached raw data buffer.
  // Uses the aliasing constructor so the returned ptr shares ownership with the
  // internal vector storage -- the cache's weak_ptr remains valid as long as
  // any consumer holds the returned ptr.
  std::shared_ptr<void> getOrInsert(const uint8_t *tensorDataPtr,
                                    const std::vector<uint32_t> &shape,
                                    const std::vector<uint32_t> &stride,
                                    uint32_t itemSize,
                                    ::tt::target::DataType dataType);

private:
  std::unordered_map<uint64_t, std::weak_ptr<std::vector<uint8_t>>> pool_;

  uint64_t getTensorHash(const uint8_t *tensorDataPtr,
                         const std::vector<uint32_t> &shape,
                         const std::vector<uint32_t> &stride, uint32_t itemSize,
                         ::tt::target::DataType dataType);
};

} // namespace tt::runtime::distributed::worker
#endif // TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_TENSOR_DATA_CACHE_H
