// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/KVCache/PagedFillCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

PagedFillCacheResolvedParams
resolvePagedFillCacheParams(const ::tt::target::ttnn::PagedFillCacheOpT &op) {
  PagedFillCacheResolvedParams params;

  return params;
}

template <typename Tag>
auto createPagedFillCacheTuple(Tag tag,
                               const ::tt::target::ttnn::PagedFillCacheOpT &op,
                               TensorArg cache, TensorArg input,
                               TensorArg pageTable,
                               std::optional<TensorArg> batchIdxTensor,
                               const PagedFillCacheResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(cache, tag), resolveTensorArg(input, tag),
      resolveTensorArg(pageTable, tag),
      batchIdxTensor
          ? std::make_optional(resolveTensorArg(*batchIdxTensor, tag))
          : std::nullopt,
      /*batch_idx=*/0,
      /*compute_kernel_config=*/std::nullopt,
      /*mesh_coords=*/std::nullopt);
}

PagedFillCacheOpResult callPagedFillCache(
    CallType callType, const ::tt::target::ttnn::PagedFillCacheOpT &op,
    TensorArg cache, TensorArg input, TensorArg pageTable,
    std::optional<TensorArg> batchIdxTensor, ::ttnn::MeshDevice *device) {
  PagedFillCacheResolvedParams params = resolvePagedFillCacheParams(op);

  auto makeTuple = [&](auto tag) {
    return createPagedFillCacheTuple(tag, op, cache, input, pageTable,
                                     batchIdxTensor, params);
  };

  return callOp<PagedFillCacheOpResult>(
      WRAP_OP(::ttnn::experimental::paged_fill_cache), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
