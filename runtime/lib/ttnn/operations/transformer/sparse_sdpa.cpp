// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/sparse_sdpa.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::SparseSdpaOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &kv = tensorPool.getTTNNTensorAndValidate(op->kv());
  const ::ttnn::Tensor &indices =
      tensorPool.getTTNNTensorAndValidate(op->indices());

  std::optional<float> scale = op->scale();

  // The output layout is fixed by the op (DRAM-interleaved ROW_MAJOR), so no
  // memory config is exposed. compute_kernel_config falls back to the ttnn
  // default; cache_batch_idx and the block-cyclic remap parameters are not
  // modelled by the TTNN dialect op and stay unset.
  ::ttnn::Tensor out = ::ttnn::transformer::sparse_sdpa(
      query, kv, indices, op->v_dim(),
      ::ttnn::transformer::SparseKVFormat::BF16, scale, op->k_chunk_size(),
      /*compute_kernel_config=*/std::nullopt,
      /*cache_batch_idx=*/std::nullopt,
      /*block_cyclic_sp_axis=*/std::nullopt,
      /*block_cyclic_chunk_local=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::transformer
