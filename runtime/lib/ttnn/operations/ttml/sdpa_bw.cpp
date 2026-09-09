// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/sdpa_bw.h"
#include "metal/ops/sdpa_bw/sdpa_bw.hpp"

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::SDPABackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &gradOutput =
      tensorPool.getTTNNTensorAndValidate(op->grad_output());
  const ::ttnn::Tensor &attnOutput =
      tensorPool.getTTNNTensorAndValidate(op->attn_output());
  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());
  const ::ttnn::Tensor &intermediates =
      tensorPool.getTTNNTensorAndValidate(op->intermediates());

  // Optional attention mask; only used when mask_type == Arbitrary.
  std::optional<::ttnn::Tensor> mask = std::nullopt;
  if (op->attention_mask()) {
    mask = tensorPool.getTTNNTensorAndValidate(op->attention_mask());
  }

  const ::ttml::metal::AttentionMaskType maskType =
      static_cast<::ttml::metal::AttentionMaskType>(op->mask_type());

  // Returns {grad_query, grad_key, grad_value}.
  auto [gradQuery, gradKey, gradValue] = ::ttml::metal::sdpa_bw(
      gradOutput, attnOutput, query, key, value, intermediates, maskType, mask,
      op->dropout_probability());

  tensorPool.insertTTNNTensorAndValidate(op->grad_query(), gradQuery);
  tensorPool.insertTTNNTensorAndValidate(op->grad_key(), gradKey);
  tensorPool.insertTTNNTensorAndValidate(op->grad_value(), gradValue);
}

} // namespace tt::runtime::ttnn::operations::ttml
