// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.hpp"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  std::optional<::ttnn::Tensor> batch_offset = std::nullopt;
  if (op->batch_offset()) {
    batch_offset = tensorPool.getTTNNTensorAndValidate(op->batch_offset());
  }

  uint32_t numHeads = op->num_heads();
  std::optional<const uint32_t> numKVHeads = op->num_kv_heads();
  std::optional<const bool> overlapQKCoregrid(op->overlap_qk_coregrid());
  std::optional<const uint32_t> sliceSize = op->slice_size();

  auto [q, k, v] = ::ttnn::experimental::nlp_create_qkv_heads_decode(
      input, numHeads, numKVHeads, overlapQKCoregrid, batch_offset, sliceSize,
      outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->q_out(), q);
  tensorPool.insertTTNNTensorAndValidate(op->k_out(), k);
  tensorPool.insertTTNNTensorAndValidate(op->v_out(), v);
}

} // namespace tt::runtime::ttnn::operations::transformer
