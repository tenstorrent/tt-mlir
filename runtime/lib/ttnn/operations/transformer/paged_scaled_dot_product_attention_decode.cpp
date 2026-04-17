// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/paged_scaled_dot_product_attention_decode.h"

#include "tt/runtime/detail/ttnn/utils.h"

#include <algorithm>

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
    const bool isBlackhole = query.device()->arch() == ::tt::ARCH::BLACKHOLE;

    // TTNN SDPA decode reduces per-head partials via a binary tree whose depth
    // is bounded by MAX_TREE_REDUCTION_ROUNDS = 6, so any schedule must keep
    // max_cores_per_head_batch <= 2^6 = 64. Blackhole's 10x11 compute grid
    // (110 cores) would otherwise trip the TT_FATAL in
    // sdpa_decode_program_factory.cpp when used directly.
    //
    // IMPORTANT: this logic is mirrored at compile time by
    //   lib/OpModel/TTNN/TTNNOpModel.cpp :: buildPagedSdpaDecodeProgramConfig
    // so that the TTNNOperationValidationAndFallback pass sees the same
    // program_config as the runtime. Keep both sites in sync when you edit
    // the schedule below.
    constexpr uint32_t kMaxSdpaDecodeCoresPerHeadBatch = 64u;

    // Estimate per-core K/V circular-buffer pressure for the default TTNN
    // causal schedule. The default path keeps resident K/V pages on every
    // core per head, so footprint scales linearly with head_dim and the
    // number of page-table blocks currently in flight. If that exceeds what
    // we are willing to reserve in per-core L1, fall back to an explicit
    // k_chunk_size=32 schedule which bounds parallelism to the number of
    // resident page-table blocks (keeping static CBs small).
    const uint32_t headDim = query.padded_shape()[-1];
    const uint32_t pageTableBlocks = pageTable.padded_shape()[-1];
    constexpr uint32_t kPageTokens = 32u;
    const uint64_t dtypeBytes = query.element_size();
    const uint64_t perPageKvBytes =
        /*K+V*/ 2u * static_cast<uint64_t>(kPageTokens) *
        static_cast<uint64_t>(headDim) * dtypeBytes *
        /*double buffer*/ 2u;
    const uint64_t perCoreL1 = query.device()->l1_size_per_core();
    // Reserve half of per-core L1 for Q/O, softmax scratch, kernel code, and
    // other overheads we do not model explicitly; give the rest to K/V
    // residency.
    const uint64_t kvBudgetPerCore = perCoreL1 / 2u;
    const bool needL1Override =
        perPageKvBytes > 0 &&
        perPageKvBytes * static_cast<uint64_t>(pageTableBlocks) >
            kvBudgetPerCore;

    const uint32_t totalCores = computeGrid.x * computeGrid.y;
    const uint32_t coresCap =
        std::min(totalCores, kMaxSdpaDecodeCoresPerHeadBatch);

    if (needL1Override) {
      // Large head_dim (e.g. Gemma-4 global_head_dim=512) or long contexts:
      // default path would over-allocate L1. Pin K/V chunks to one page so
      // static CBs stay small, and cap cores per head batch by the number of
      // page-table blocks in the current request (further capped by the tree
      // reduction limit).
      programConfig.emplace();
      programConfig->q_chunk_size = 0;
      programConfig->k_chunk_size = kPageTokens;
      programConfig->compute_with_storage_grid_size = computeGrid;
      programConfig->max_cores_per_head_batch =
          std::min(pageTableBlocks, coresCap);
    } else if (isBlackhole) {
      // Preserve the pre-existing Blackhole causal schedule
      // (k_chunk_size=0) with as many cores per head as the tree reduction
      // allows. Required for current BH performance on small head dimensions
      // where the default TTNN path fits L1 easily but still needs the
      // exp_approx_mode workaround below.
      programConfig.emplace();
      programConfig->q_chunk_size = 0;
      programConfig->k_chunk_size = 0;
      programConfig->compute_with_storage_grid_size = computeGrid;
      programConfig->max_cores_per_head_batch = coresCap;
    }
    // else: non-BH arch with L1 headroom. Leave programConfig unset so TTNN
    // picks its own default schedule.

    if (isBlackhole) {
      if (!programConfig.has_value()) {
        programConfig.emplace();
        programConfig->compute_with_storage_grid_size = computeGrid;
      }
      // Blackhole-only workaround: the approximate exponential path is
      // unreliable on current silicon for causal decode.
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
