// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_PROGRAM_DESC_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_PROGRAM_DESC_CACHE_H

#include "tt/runtime/detail/ttnn/ttnn.h"

namespace tt::runtime::ttnn {

class ProgramDescCache {
public:
  ProgramDescCache() {};

  ProgramDescCache(const ProgramDescCache &) = delete;
  ProgramDescCache &operator=(const ProgramDescCache &) = delete;

  ProgramDescCache(ProgramDescCache &&) = delete;
  ProgramDescCache &operator=(ProgramDescCache &&) = delete;

  const ::tt::tt_metal::ProgramDescriptor *
  get(const ::tt::target::ttnn::ProgramDescriptor *programDesc) const;

  void insert(const ::tt::target::ttnn::ProgramDescriptor *programDesc,
              const ::tt::tt_metal::ProgramDescriptor &programDescriptor);

private:
  std::unordered_map<const ::tt::target::ttnn::ProgramDescriptor *,
                     ::tt::tt_metal::ProgramDescriptor>
      cache_;
};
} // namespace tt::runtime::ttnn

#endif
