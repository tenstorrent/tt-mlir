// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/chunked_scaled_dot_product_attention.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runChunkedScaledDotProductAttentionOp(
    const ::tt::target::ttnn::ChunkedScaledDotProductAttentionOp *op,
    ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());
  const ::ttnn::Tensor &pageTable =
      tensorPool.getTTNNTensorAndValidate(op->page_table());
  // Prefix offset lives in a device tensor (the "flexible" overload) so the op
  // is trace-compatible: the offset is read at runtime, no recompile per chunk.
  const ::ttnn::Tensor &chunkStartIdx =
      tensorPool.getTTNNTensorAndValidate(op->chunk_start_idx());

  std::optional<float> scale = op->scale();

  // Like the (prefill) scaled_dot_product_attention runtime, leave the program
  // config null unless one was provided: ttnn computes valid q/k chunk sizes
  // from the shapes. Forcing q/k_chunk_size=0 here divides-by-zero in the
  // prefill SDPAProgramFactory (unlike the decode kernel, which accepts
  // 0=auto).
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig = std::nullopt;
  if (op->program_config()) {
    programConfig = utils::createSDPAProgramConfig(op->program_config());
  }

  ::ttnn::Tensor out =
      ::ttnn::transformer::chunked_scaled_dot_product_attention(
          query, key, value, pageTable, chunkStartIdx, scale,
          outputMemoryConfig,
          /*program_config=*/programConfig,
          /*compute_kernel_config=*/std::nullopt);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::ChunkedScaledDotProductAttentionOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runChunkedScaledDotProductAttentionOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
