// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_KVCACHE_PAGEDFILLCACHEOP_H
#define TTMLIR_OPINVOKE_TTNN_KVCACHE_PAGEDFILLCACHEOP_H

#include "operations/experimental/paged_cache/paged_cache.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using PagedFillCacheOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PagedFillCacheResolvedParams {};

PagedFillCacheResolvedParams
resolvePagedFillCacheParams(const ::tt::target::ttnn::PagedFillCacheOpT &op);

PagedFillCacheOpResult
callPagedFillCache(CallType callType,
                   const ::tt::target::ttnn::PagedFillCacheOpT &op,
                   TensorArg cache, TensorArg input, TensorArg pageTable,
                   std::optional<TensorArg> batchIdxTensor,
                   ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_KVCACHE_PAGEDFILLCACHEOP_H
