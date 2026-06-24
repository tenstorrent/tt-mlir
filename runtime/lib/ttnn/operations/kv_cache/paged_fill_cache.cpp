// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/paged_fill_cache.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/workarounds.h"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::PagedFillCacheOp *op,
         ProgramContext &context) {

  const ::ttnn::Tensor &cacheTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &inputTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &pageTableTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->page_table());

  const std::optional<::ttnn::Tensor> &batchIdxTensor =
      op->batch_idx_tensor()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->batch_idx_tensor()))
          : std::nullopt;

  // Force the MeshWorkloadFactory by passing the cache tensor's full set of
  // mesh coordinates. With std::nullopt, the device op selects the
  // single-program factory, which under DP partitioning runs the kernel on
  // the mesh root only - so per-rank cache writes from non-root devices are
  // silently dropped.
  //
  // TODO(#47955): ttnn could infer these coords from the cache tensor (it is
  // already an op input) and select the MeshWorkloadFactory by default, letting
  // us drop this explicit pass. Tracked in tenstorrent/tt-metal#47955.
  const auto &coordVec = cacheTensor.tensor_topology().mesh_coords();
  std::set<::ttnn::MeshCoordinate> meshCoords(coordVec.begin(), coordVec.end());

  ::ttnn::experimental::paged_fill_cache(cacheTensor, inputTensor,
                                         pageTableTensor, batchIdxTensor,
                                         /*batch_offset=*/0,
                                         /*compute_kernel_config*/ std::nullopt,
                                         /*mesh_coords*/ meshCoords);
}
} // namespace tt::runtime::ttnn::operations::kv_cache
