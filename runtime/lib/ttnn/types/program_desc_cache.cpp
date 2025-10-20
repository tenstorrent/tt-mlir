// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/program_desc_cache.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn {

const ::tt::tt_metal::ProgramDescriptor *ProgramDescCache::get(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc) const {
  auto it = cache_.find(programDesc);
  if (it != cache_.end()) {
    return &it->second;
  }
  return nullptr;
}

void ProgramDescCache::insert(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc,
    const ::tt::tt_metal::ProgramDescriptor &programDescriptor) {
  cache_.try_emplace(programDesc, programDescriptor);
}

} // namespace tt::runtime::ttnn
