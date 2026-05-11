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
  std::optional<uint32_t> slidingWindowSize = op->sliding_window_size();
  auto computeGrid = query.device()->compute_with_storage_grid_size();
  if (op->core_grid()) {
    computeGrid = ::tt::runtime::ttnn::utils::toTTNNCoreCoord(*op->core_grid());
  }

  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig = std::nullopt;
  if (!isCausal) {
    programConfig.emplace();
    programConfig->k_chunk_size = 32; // Required for non-causal
    programConfig->compute_with_storage_grid_size = computeGrid;
  } else {
    const bool isBlackhole = query.device()->arch() == ::tt::ARCH::BLACKHOLE;

    // tt-metal's SDPA decode program factory allocates the following static
    // per-core CBs (see sdpa_decode_program_factory.cpp):
    //
    //   k_tiles                = Sk_chunk_t_cb_size * DHt  * 2  // double-buffered
    //   v_tiles                = Sk_chunk_t_cb_size * vDHt * 2  // double-buffered
    //   intermed_output_tiles  = (PNHt * vDHt + 2 * PNHt) * (num_cores_per_head - 1)
    //
    // where DHt = head_dim / TILE_WIDTH. num_cores_per_head is the
    // max_cores_per_head_batch chosen by the schedule, bounded by
    // MAX_TREE_REDUCTION_ROUNDS = 6 (i.e. 2^6 = 64 cores per head).
    //
    // Both K/V CBs and the intermed-output CB scale linearly with head_dim,
    // and the intermed-output CB also scales with num_cores_per_head. For
    // large head_dim (Gemma-4 31B uses 512) this overflows per-core L1 under
    // the default schedule, even on Blackhole's 1.5 MB usable L1. We trigger
    // an override schedule for those cases: pin k_chunk_size=32 (one page)
    // to bound Sk_chunk_t_cb_size, and cap max_cores_per_head_batch at 32
    // (vs the 64-core tree-reduction limit) to halve the intermed-output
    // footprint.
    //
    // For Llama-class head_dim (128) the default schedule fits comfortably,
    // so we leave it alone. Gemma-4's sliding-window layers use head_dim=256
    // and still need the override -- the intermed-output CB at 64 cores per
    // head is the bottleneck even when K/V residency is moderate.
    //
    // IMPORTANT: this logic is mirrored at compile time by
    //   lib/OpModel/TTNN/TTNNOpModel.cpp :: buildPagedSdpaDecodeProgramConfig
    // so the TTNNOperationValidationAndFallback pass sees the same
    // program_config the runtime will execute. Keep both sites in sync.
    constexpr uint32_t kMaxSdpaDecodeCoresPerHeadBatch = 64u;
    constexpr uint32_t kPageTokens = 32u;
    constexpr uint32_t kOverrideHeadDimThreshold = 256u;
    constexpr uint32_t kOverrideCoresCap = 32u;

    const uint32_t headDim = query.padded_shape()[-1];
    const uint32_t pageTableBlocks = pageTable.padded_shape()[-1];
    const bool needL1Override = headDim >= kOverrideHeadDimThreshold;

    const uint32_t totalCores = computeGrid.x * computeGrid.y;
    const uint32_t coresCap =
        std::min(totalCores, kMaxSdpaDecodeCoresPerHeadBatch);

    if (needL1Override) {
      const uint32_t overrideCap = std::min(coresCap, kOverrideCoresCap);
      programConfig.emplace();
      programConfig->q_chunk_size = 0;
      programConfig->k_chunk_size = kPageTokens;
      programConfig->compute_with_storage_grid_size = computeGrid;
      programConfig->max_cores_per_head_batch =
          std::min(pageTableBlocks, overrideCap);
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
