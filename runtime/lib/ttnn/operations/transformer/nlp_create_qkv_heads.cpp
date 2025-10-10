// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::NLPCreateQKVHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &inputQ =
      tensorPool.getTTNNTensorAndValidate(op->input_q());

  std::optional<::ttnn::Tensor> inputKV = std::nullopt;
  if (op->input_kv()) {
    inputKV = tensorPool.getTTNNTensorAndValidate(op->input_kv());
  }

  uint32_t numQHeads = op->num_q_heads();
  std::optional<uint32_t> numKVHeads = std::nullopt;
  if (op->num_kv_heads() != 0) {
    numKVHeads = op->num_kv_heads();
  }
  bool transposeKHeads = op->transpose_k_heads();

  auto [query, key, value] = ::ttnn::experimental::nlp_create_qkv_heads(
      inputQ, inputKV, numQHeads, numKVHeads, transposeKHeads,
      outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->query(), query);
  tensorPool.insertTTNNTensorAndValidate(op->key(), key);
  tensorPool.insertTTNNTensorAndValidate(op->value(), value);
}

} // namespace tt::runtime::ttnn::operations::transformer
