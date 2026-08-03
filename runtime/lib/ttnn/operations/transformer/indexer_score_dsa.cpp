// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/indexer_score_dsa.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::IndexerScoreDsaOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &weights =
      tensorPool.getTTNNTensorAndValidate(op->weights());

  // Mesh axis carrying the query sequence shard. Unset leaves the op on its
  // flat row-major enumeration over all of q's devices, which is only correct
  // when the sequence is sharded across every device -- naming the axis is what
  // makes a partial split (e.g. heads on one axis, sequence on another)
  // correct.
  std::optional<std::vector<uint32_t>> seqShardAxes = std::nullopt;
  if (op->cluster_axis()) {
    seqShardAxes = std::vector<uint32_t>{*op->cluster_axis()};
  }

  // program_config and compute_kernel_config fall back to the ttnn defaults.
  ::ttnn::Tensor out = ::ttnn::experimental::indexer_score_dsa(
      query, key, weights, op->chunk_start_idx(),
      /*program_config=*/{}, /*compute_kernel_config=*/std::nullopt,
      /*cache_batch_idx=*/std::nullopt, /*kv_len=*/std::nullopt, seqShardAxes);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::transformer
