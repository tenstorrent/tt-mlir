// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/sdpa_fw.h"
#include "metal/common/const_utils.hpp"  // ttml::metal::AttentionMaskType
#include "metal/ops/sdpa_fw/sdpa_fw.hpp" // ttml::metal::sdpa_fw

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::SDPAForwardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());

  // Optional attention mask; only used when mask_type == Arbitrary.
  std::optional<::ttnn::Tensor> mask = std::nullopt;
  if (op->attention_mask()) {
    mask = tensorPool.getTTNNTensorAndValidate(op->attention_mask());
  }

  const ::ttml::metal::AttentionMaskType maskType =
      static_cast<::ttml::metal::AttentionMaskType>(op->mask_type());

  // Returns {output, intermediate-or-nullopt}.
  std::vector<std::optional<::ttnn::Tensor>> result = ::ttml::metal::sdpa_fw(
      query, key, value, maskType, mask, op->dropout_probability(),
      op->return_intermediates());

  tensorPool.insertTTNNTensorAndValidate(op->out(), result.at(0).value());

  if (op->return_intermediates() && op->intermediates()) {
    tensorPool.insertTTNNTensorAndValidate(op->intermediates(),
                                           result.at(1).value());
  }
}

} // namespace tt::runtime::ttnn::operations::ttml
