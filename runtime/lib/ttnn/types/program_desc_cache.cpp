// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/program_desc_cache.h"

namespace tt::runtime::ttnn {

void *ProgramDescCache::get(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc) const {
  auto it = cache_.find(programDesc);
  if (it != cache_.end()) {
    return it->second.get();
  }
  return nullptr;
}

void ProgramDescCache::insert(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc,
    std::shared_ptr<void> programDescriptor) {
  cache_.try_emplace(programDesc, std::move(programDescriptor));
}

} // namespace tt::runtime::ttnn
