// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/KVCache/PagedFillCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

PagedFillCacheResolvedParams
resolvePagedFillCacheParams(const ::tt::target::ttnn::PagedFillCacheOpT &op,
                            CallType callType, TensorArg cache) {
  PagedFillCacheResolvedParams params;

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
      /*mesh_coords=*/params.meshCoords);
}

PagedFillCacheOpResult callPagedFillCache(
    CallType callType, const ::tt::target::ttnn::PagedFillCacheOpT &op,
    TensorArg cache, TensorArg input, TensorArg pageTable,
    std::optional<TensorArg> batchIdxTensor, ::ttnn::MeshDevice *device) {
  PagedFillCacheResolvedParams params =
      resolvePagedFillCacheParams(op, callType, cache);

  auto makeTuple = [&](auto tag) {
    return createPagedFillCacheTuple(tag, op, cache, input, pageTable,
                                     batchIdxTensor, params);
  };

  return callOp<PagedFillCacheOpResult>(
      WRAP_OP(::ttnn::experimental::paged_fill_cache), callType, makeTuple,
      device);
}

} // namespace ttnn_op_invoke
