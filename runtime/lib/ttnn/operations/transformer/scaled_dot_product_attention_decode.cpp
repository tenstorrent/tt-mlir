// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/scaled_dot_product_attention_decode.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runScaledDotProductAttentionDecodeOp(
    const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOp *op,
    ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());
  bool isCausal = op->is_causal();
  const std::optional<::ttnn::Tensor> &attentionMask =
      op->attention_mask()
          ? std::make_optional(
                tensorPool.getTTNNTensorAndValidate(op->attention_mask()))
          : std::nullopt;
  const std::optional<::ttnn::Tensor> &curPosTensor =
      op->cur_pos_tensor()
          ? std::make_optional(
                tensorPool.getTTNNTensorAndValidate(op->cur_pos_tensor()))
          : std::nullopt;
  const std::optional<::ttnn::Tensor> &attentionSink =
      op->attention_sink()
          ? std::make_optional(
                tensorPool.getTTNNTensorAndValidate(op->attention_sink()))
          : std::nullopt;
  float scale = op->scale();

  // Must pass a uint vector for curPos even though we are using a tensor
  // instead, as the TTNN API expects a vector.
  constexpr std::vector<uint32_t> curPosEmpty = {};
  ::ttnn::Tensor out = ::ttnn::transformer::scaled_dot_product_attention_decode(
      query, key, value, isCausal, attentionMask, curPosEmpty, curPosTensor,
      attentionSink, scale, outputMemoryConfig, std::nullopt, std::nullopt);
  auto onHost = ::ttnn::from_device(out);
  onHost.print();
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::ScaledDotProductAttentionDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runScaledDotProductAttentionDecodeOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
