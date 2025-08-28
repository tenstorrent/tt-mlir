// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/attention/paged_attention.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/types.h"
#include <operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <string_view>
#include <vector>

namespace tt::runtime::ttnn::operations::attention {

void run(const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp *op,
         ProgramContext &context) {

  const ::ttnn::Tensor &query =
      context.getTensorPool().getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &keys =
      context.getTensorPool().getTTNNTensorAndValidate(op->keys());
  const ::ttnn::Tensor &values =
      context.getTensorPool().getTTNNTensorAndValidate(op->values());
  const ::ttnn::Tensor &page_table =
      context.getTensorPool().getTTNNTensorAndValidate(op->page_table());
  const std::optional<::ttnn::Tensor> &attn_mask =
      op->attn_mask()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->attn_mask()))
          : std::nullopt;
  const std::optional<::ttnn::Tensor> &cur_pos_tensor =
      op->cur_pos_tensor()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->cur_pos_tensor()))
          : std::nullopt;

  ::ttnn::Tensor output =
      ::ttnn::transformer::paged_scaled_dot_product_attention_decode(
          query, keys, values, page_table, op->is_causal(), attn_mask,
          cur_pos_tensor, std::nullopt, op->scale());

  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::attention
