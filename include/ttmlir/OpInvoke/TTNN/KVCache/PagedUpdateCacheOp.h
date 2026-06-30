// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_KVCACHE_PAGEDUPDATECACHEOP_H
#define TTMLIR_OPINVOKE_TTNN_KVCACHE_PAGEDUPDATECACHEOP_H

#include "operations/experimental/paged_cache/paged_cache.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using PagedUpdateCacheOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PagedUpdateCacheResolvedParams {
  std::vector<uint32_t> update_idxs;
  std::optional<std::set<::ttnn::MeshCoordinate>> meshCoords;
};

PagedUpdateCacheResolvedParams
resolvePagedUpdateCacheParams(const ::tt::target::ttnn::PagedUpdateCacheOpT &op,
                              CallType callType, TensorArg cache);

PagedUpdateCacheOpResult callPagedUpdateCache(
    CallType callType, const ::tt::target::ttnn::PagedUpdateCacheOpT &op,
    TensorArg cache, TensorArg input, std::optional<TensorArg> updateIndex,
    std::optional<TensorArg> pageTable, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_KVCACHE_PAGEDUPDATECACHEOP_H
