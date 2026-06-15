// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_KVCACHE_FILLCACHEOP_H
#define TTMLIR_OPINVOKE_TTNN_KVCACHE_FILLCACHEOP_H

#include "operations/kv_cache/kv_cache.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn_op_invoke {

using FillCacheOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct FillCacheResolvedParams {};

FillCacheResolvedParams
resolveFillCacheParams(const ::tt::target::ttnn::FillCacheOpT &op);

FillCacheOpResult callFillCache(CallType callType,
                                const ::tt::target::ttnn::FillCacheOpT &op,
                                TensorArg cache, TensorArg input,
                                ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_KVCACHE_FILLCACHEOP_H
