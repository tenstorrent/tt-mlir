// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_PROGRAM_DESC_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_PROGRAM_DESC_CACHE_H

#include "ttmlir/Target/TTNN/Target.h"

#include <unordered_map>

namespace tt::runtime {

class ProgramDescCache {
public:
  ProgramDescCache() {};

  ProgramDescCache(const ProgramDescCache &) = delete;
  ProgramDescCache &operator=(const ProgramDescCache &) = delete;

  ProgramDescCache(ProgramDescCache &&) = delete;
  ProgramDescCache &operator=(ProgramDescCache &&) = delete;

  void *get(const std::size_t &hash) const {
    auto it = cache_.find(hash);
    if (it != cache_.end()) {
      return it->second.get();
    }
    return nullptr;
  }
  void insert(const std::size_t &hash,
              std::shared_ptr<void> programDescriptor) {
    cache_.try_emplace(hash, std::move(programDescriptor));
  }

private:
  std::unordered_map<std::size_t, std::shared_ptr<void>> cache_;
};
} // namespace tt::runtime

#endif
