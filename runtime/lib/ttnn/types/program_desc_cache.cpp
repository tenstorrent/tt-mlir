// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/program_desc_cache.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn {

const ::tt::tt_metal::ProgramDescriptor *ProgramDescCache::get(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc) const {
  auto it = std::find_if(
      cache_.begin(), cache_.end(),
      [programDesc](const auto &pair) { return pair.first == programDesc; });
  if (it != cache_.end()) {
    return &it->second;
  }
  return nullptr;
}

void ProgramDescCache::insert(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc,
    const ::tt::tt_metal::ProgramDescriptor &programDescriptor) {
  if (cache_.size() >= MAX_SIZE) {
    LOG_WARNING("ProgramDescCache is full.");
    cache_.erase(cache_.begin());
  }
  cache_.emplace_back(programDesc, programDescriptor);
}

} // namespace tt::runtime::ttnn
