// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/scaled_dot_product_attention.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runScaledDotProductAttentionOp(
    const ::tt::target::ttnn::ScaledDotProductAttentionOp *op,
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

  std::optional<float> scale = op->scale();

  ::ttnn::Tensor out = ::ttnn::transformer::scaled_dot_product_attention(
      query, key, value, attentionMask, isCausal, scale, outputMemoryConfig,
      std::nullopt, std::nullopt);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::ScaledDotProductAttentionOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runScaledDotProductAttentionOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
