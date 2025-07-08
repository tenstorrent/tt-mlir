// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_CACHE_LOAD_CACHED_H
#define TT_RUNTIME_TTNN_OPERATIONS_CACHE_LOAD_CACHED_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::operations::cache {

void run(const ::tt::target::ttnn::LoadCachedOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::cache

#endif // TT_RUNTIME_TTNN_OPERATIONS_CACHE_LOAD_CACHED_H
