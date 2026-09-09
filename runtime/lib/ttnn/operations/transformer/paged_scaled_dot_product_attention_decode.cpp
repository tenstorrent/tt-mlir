// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/paged_scaled_dot_product_attention_decode.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
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

  // The SDPAProgramConfig is populated in TTNN IR by the workaround pass
  // (PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern).
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig = std::nullopt;
  if (op->program_config()) {
    programConfig = utils::createSDPAProgramConfig(op->program_config());
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
