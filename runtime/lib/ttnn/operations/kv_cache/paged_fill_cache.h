// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_PAGED_FILL_CACHE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_PAGED_FILL_CACHE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::PagedFillCacheOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::kv_cache

#endif
