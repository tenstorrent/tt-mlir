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
  } else if (query.device()->arch() == ::tt::ARCH::BLACKHOLE) {
    // Preserve the existing causal decode scheduling while disabling the
    // approximate exponential path on Blackhole.
    programConfig.emplace();
    programConfig->q_chunk_size = 0;
    programConfig->k_chunk_size = 0;

    // TODO: Remove grid capping once tt-metal fixes core allocation
    // (https://github.com/tenstorrent/tt-metal/issues/40978)
    // On Blackhole (10x11=110 cores), batch sizes like 32 with nkv=8 cause
    // bad integer division in sdpa_decode_program_factory: 110/32=3
    // cores_per_batch, ceil(8/3)=3 heads_per_core, but 8%3!=0. Cap to 8x8
    // which gives clean division for all common num_kv_heads values.
    auto safeGrid = computeGrid;
    uint32_t B = pageTable.padded_shape()[0];
    uint32_t numKVHeads = key.padded_shape()[1];
    uint32_t numCores = safeGrid.x * safeGrid.y;
    uint32_t coresPerBatchUncapped = numCores / B;
    if (coresPerBatchUncapped > 0) {
      uint32_t headsPerCore =
          (numKVHeads + coresPerBatchUncapped - 1) / coresPerBatchUncapped;
      if (numKVHeads % headsPerCore != 0) {
        safeGrid = {std::min<size_t>(computeGrid.x, 8),
                    std::min<size_t>(computeGrid.y, 8)};
      }
    }

    programConfig->compute_with_storage_grid_size = safeGrid;
    programConfig->max_cores_per_head_batch = safeGrid.x * safeGrid.y;
    programConfig->exp_approx_mode = false;
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
