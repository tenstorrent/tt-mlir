// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/paged_scaled_dot_product_attention_decode.h"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runPagedScaledDotProductAttentionDecodeOp(
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp *op,
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
  bool isCausal = op->is_causal();

  std::optional<::ttnn::Tensor> attentionMask = std::nullopt;
  if (op->attention_mask()) {
    attentionMask.emplace(
        tensorPool.getTTNNTensorAndValidate(op->attention_mask()));
  }

  std::optional<::ttnn::Tensor> curPosTensor = std::nullopt;
  if (op->cur_pos_tensor()) {
    curPosTensor.emplace(
        tensorPool.getTTNNTensorAndValidate(op->cur_pos_tensor()));
  }

  std::optional<::ttnn::Tensor> attentionSink = std::nullopt;
  if (op->attention_sink()) {
    attentionSink.emplace(
        tensorPool.getTTNNTensorAndValidate(op->attention_sink()));
  }

  std::optional<float> scale = op->scale();
  std::optional<uint32_t> slidingWindowSize = std::nullopt;
  const auto computeGrid = query.device()->compute_with_storage_grid_size();

  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig = std::nullopt;
  if (!isCausal) {
    programConfig.emplace();
    programConfig->k_chunk_size = 32; // Required for non-causal
    programConfig->compute_with_storage_grid_size = computeGrid;
  } else {
    programConfig.emplace();
    programConfig->q_chunk_size = 0;
    // Use one page (32 tokens) per K/V chunk for causal paged decode. This
    // keeps the static K/V circular buffers small while still allowing
    // parallelism to scale with the number of resident page-table blocks.
    programConfig->k_chunk_size = 32;
    programConfig->compute_with_storage_grid_size = computeGrid;
    // Causal paged decode falls back to using every core per head when no
    // program config is provided, which can over-allocate L1 for large head
    // dimensions. With 32-token K/V chunks, the useful parallelism per head is
    // bounded by the number of page-table blocks in the current request.
    const uint32_t totalCores = computeGrid.x * computeGrid.y;
    const uint32_t pageTableBlocks = pageTable.padded_shape()[-1];
    programConfig->max_cores_per_head_batch =
        pageTableBlocks < totalCores ? pageTableBlocks : totalCores;
    if (query.device()->arch() == ::tt::ARCH::BLACKHOLE) {
      // Preserve the existing Blackhole workaround that disables the
      // approximate exponential path for causal decode.
      programConfig->exp_approx_mode = false;
    }
  }

  ::ttnn::Tensor out =
      ::ttnn::transformer::paged_scaled_dot_product_attention_decode(
          query, key, value, pageTable, isCausal, attentionMask, curPosTensor,
          attentionSink, scale, slidingWindowSize, outputMemoryConfig,
          /*program_config=*/programConfig,
          /*compute_kernel_config=*/std::nullopt);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runPagedScaledDotProductAttentionDecodeOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
