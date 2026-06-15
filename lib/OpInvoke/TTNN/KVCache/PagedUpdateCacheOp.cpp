// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/KVCache/PagedUpdateCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include <optional>

namespace ttnn_op_invoke {

PagedUpdateCacheResolvedParams
resolvePagedUpdateCacheParams(const ::tt::target::ttnn::PagedUpdateCacheOpT &op,
                              CallType callType, TensorArg cache) {
  PagedUpdateCacheResolvedParams params;

  params.update_idxs = {};

  // TODO: This branching on callType should be removed.
  if (callType == CallType::EXECUTE) {
    // Force the MeshWorkloadFactory by passing the cache tensor's full set of
    // mesh coordinates. With std::nullopt, the device op selects the
    // single-program factory, which under DP partitioning runs the kernel on
    // the mesh root only - so per-rank cache writes from non-root devices are
    // silently dropped. Per-rank divergent updates require one program per
    // mesh coordinate, which is what the MeshWorkloadFactory provides.
    //
    // TODO(#47955): ttnn could infer these coords from the cache tensor (it is
    // already an op input) and select the MeshWorkloadFactory by default,
    // letting us drop this explicit pass. Tracked in
    // tenstorrent/tt-metal#47955.
    ::ttnn::Tensor cacheTensor = *std::get<const ::ttnn::Tensor *>(cache);
    const auto &coordVec = cacheTensor.tensor_topology().mesh_coords();
    std::set<::ttnn::MeshCoordinate> meshCoords(coordVec.begin(),
                                                coordVec.end());
    params.meshCoords = std::make_optional(meshCoords);
  } else {
    params.meshCoords = std::nullopt;
  }

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
      /*mesh_coords=*/params.meshCoords);
}

PagedUpdateCacheOpResult callPagedUpdateCache(
    CallType callType, const ::tt::target::ttnn::PagedUpdateCacheOpT &op,
    TensorArg cache, TensorArg input, std::optional<TensorArg> updateIndex,
    std::optional<TensorArg> pageTable, ::ttnn::MeshDevice *device) {
  PagedUpdateCacheResolvedParams params =
      resolvePagedUpdateCacheParams(op, callType, cache);

  auto makeTuple = [&](auto tag) {
    return createPagedUpdateCacheTuple(tag, op, cache, input, updateIndex,
                                       pageTable, params);
  };

  return callOp<PagedUpdateCacheOpResult>(
      WRAP_OP(::ttnn::experimental::paged_update_cache), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
