// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/paged_fill_cache.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/workarounds.h"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace tt::runtime::ttnn::operations::kv_cache {

// Tier 2B batched fill: the graph emits a SINGLE paged_fill_cache whose input is
// [num_users, num_heads, seq, head_dim] and whose batch_idx_tensor is [num_users].
// The multi-user TTNN/metal kernel fills all users in one dispatch (work is laid
// out user-major; input row u -> page_table row u), so the runtime just forwards
// the batched tensors directly. num_users == 1 is the legacy single-user fill.
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

  ::ttnn::experimental::paged_fill_cache(cacheTensor, inputTensor,
                                         pageTableTensor, batchIdxTensor,
                                         /*batch_offset=*/0,
                                         /*compute_kernel_config*/ std::nullopt,
                                         /*mesh_coords*/ std::nullopt);
}
} // namespace tt::runtime::ttnn::operations::kv_cache
