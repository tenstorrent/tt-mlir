// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/paged_fill_cache.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/workarounds.h"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace tt::runtime::ttnn::operations::kv_cache {

// Tier 2A batched fill: the graph emits a SINGLE paged_fill_cache whose input
// is [num_users, num_heads, seq, head_dim] and whose batch_idx_tensor is
// [num_users] (one page-table row per user). The underlying TTNN/metal kernel
// fills a single user per call (input [1, num_heads, seq, head_dim], a
// 1-element batch index), so we loop here, slicing one user's input + batch
// index each iteration and filling the cache in place. This keeps the compiled
// graph flat (one op per layer) instead of the per-user Python loop that
// unrolled under torch.export into thousands of ops + lifted constants
// (issue #5032). Tier 2B would replace this loop with a true multi-user kernel.
void run(const ::tt::target::ttnn::PagedFillCacheOp *op,
         ProgramContext &context) {

  const ::ttnn::Tensor &cacheTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &inputTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &pageTableTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->page_table());

  const std::optional<::ttnn::Tensor> batchIdxTensor =
      op->batch_idx_tensor()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->batch_idx_tensor()))
          : std::nullopt;

  const auto &inputShape = inputTensor.logical_shape();
  const int32_t numUsers = static_cast<int32_t>(inputShape[0]);
  const int32_t numHeads = static_cast<int32_t>(inputShape[1]);
  const int32_t seqLen = static_cast<int32_t>(inputShape[2]);
  const int32_t headDim = static_cast<int32_t>(inputShape[3]);

  for (int32_t user = 0; user < numUsers; ++user) {
    // Slice this user's KV: input[user:user+1, :, :, :] -> [1, H, seq, D].
    ::ttsl::SmallVector<int32_t> inBegins = {user, 0, 0, 0};
    ::ttsl::SmallVector<int32_t> inEnds = {user + 1, numHeads, seqLen, headDim};
    ::ttsl::SmallVector<int32_t> inStep = {1, 1, 1, 1};
    ::ttnn::Tensor inputSlice = ::ttnn::slice(
        inputTensor, ::ttsl::Span<const int32_t>(inBegins.data(),
                                                 inBegins.size()),
        ::ttsl::Span<const int32_t>(inEnds.data(), inEnds.size()),
        ::ttsl::Span<const int32_t>(inStep.data(), inStep.size()),
        /*memory_config=*/std::nullopt);

    // Slice this user's page-table row index: batch_idx[user:user+1] -> [1].
    std::optional<::ttnn::Tensor> batchIdxSlice = std::nullopt;
    if (batchIdxTensor) {
      ::ttsl::SmallVector<int32_t> idxBegins = {user};
      ::ttsl::SmallVector<int32_t> idxEnds = {user + 1};
      ::ttsl::SmallVector<int32_t> idxStep = {1};
      batchIdxSlice = ::ttnn::slice(
          *batchIdxTensor,
          ::ttsl::Span<const int32_t>(idxBegins.data(), idxBegins.size()),
          ::ttsl::Span<const int32_t>(idxEnds.data(), idxEnds.size()),
          ::ttsl::Span<const int32_t>(idxStep.data(), idxStep.size()),
          /*memory_config=*/std::nullopt);
    }

    // Fills the cache in place for this user; iterations write disjoint blocks.
    ::ttnn::experimental::paged_fill_cache(cacheTensor, inputSlice,
                                           pageTableTensor, batchIdxSlice,
                                           /*batch_idx=*/0,
                                           /*compute_kernel_config*/ std::nullopt,
                                           /*mesh_coords*/ std::nullopt);
  }
}
} // namespace tt::runtime::ttnn::operations::kv_cache
