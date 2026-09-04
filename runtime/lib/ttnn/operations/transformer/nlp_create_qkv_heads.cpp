// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/nlp_create_qkv_heads.h"

#include "operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"

#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::NLPCreateQKVHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  std::optional<::ttnn::Tensor> inputKV = std::nullopt;
  if (op->kv_input()) {
    inputKV = tensorPool.getTTNNTensorAndValidate(op->kv_input());
  }

  uint32_t numQHeads = op->num_q_heads();
  std::optional<uint32_t> numKVHeads;
  if (op->num_kv_heads()) {
    numKVHeads = op->num_kv_heads();
  }

  auto [q, k, v] = ::ttnn::experimental::nlp_create_qkv_heads(
      input, inputKV, numQHeads, numKVHeads, op->transpose_k_heads(),
      outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->q_out(), q);
  tensorPool.insertTTNNTensorAndValidate(op->k_out(), k);
  tensorPool.insertTTNNTensorAndValidate(op->v_out(), v);
}

} // namespace tt::runtime::ttnn::operations::transformer
