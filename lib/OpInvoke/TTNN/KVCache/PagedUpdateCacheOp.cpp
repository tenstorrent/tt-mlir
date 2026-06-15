// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/KVCache/PagedUpdateCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

PagedUpdateCacheResolvedParams resolvePagedUpdateCacheParams(
    const ::tt::target::ttnn::PagedUpdateCacheOpT &op) {
  PagedUpdateCacheResolvedParams params;

  params.update_idxs = {};

  return params;
}

template <typename Tag>
auto createPagedUpdateCacheTuple(
    Tag tag, const ::tt::target::ttnn::PagedUpdateCacheOpT &op, TensorArg cache,
    TensorArg input, std::optional<TensorArg> updateIndex,
    std::optional<TensorArg> pageTable,
    const PagedUpdateCacheResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(cache, tag), resolveTensorArg(input, tag),
      params.update_idxs,
      updateIndex ? std::make_optional(resolveTensorArg(*updateIndex, tag))
                  : std::nullopt,
      op.share_cache,
      pageTable ? std::make_optional(resolveTensorArg(*pageTable, tag))
                : std::nullopt,
      /*batch_offset=*/0,
      /*compute_kernel_config=*/std::nullopt,
      /*mesh_coords=*/std::nullopt);
}

PagedUpdateCacheOpResult callPagedUpdateCache(
    CallType callType, const ::tt::target::ttnn::PagedUpdateCacheOpT &op,
    TensorArg cache, TensorArg input, std::optional<TensorArg> updateIndex,
    std::optional<TensorArg> pageTable, ::ttnn::MeshDevice *device) {
  PagedUpdateCacheResolvedParams params = resolvePagedUpdateCacheParams(op);

  auto makeTuple = [&](auto tag) {
    return createPagedUpdateCacheTuple(tag, op, cache, input, updateIndex,
                                       pageTable, params);
  };

  return callOp<PagedUpdateCacheOpResult>(
      WRAP_OP(::ttnn::experimental::paged_update_cache), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
