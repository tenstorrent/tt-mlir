// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/split_query_key_value_and_split_heads.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runSplitQueryKeyValueAndSplitHeadsOp(
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOp *op,
    ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  uint32_t numHeads = op->num_heads();
  std::optional<uint32_t> numKVHeads;
  if (op->num_kv_heads()) {
    numKVHeads = op->num_kv_heads();
  }
  std::optional<::ttnn::Tensor> kvInputTensor =
      op->kv_input() ? std::make_optional(
                           tensorPool.getTTNNTensorAndValidate(op->kv_input()))
                     : std::nullopt;

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  bool transposeKey = op->transpose_key();

  auto [q, k, v] = ::ttnn::transformer::split_query_key_value_and_split_heads(
      in, kvInputTensor, numHeads, numKVHeads, transposeKey,
      outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->q_out(), q);
  tensorPool.insertTTNNTensorAndValidate(op->k_out(), k);
  tensorPool.insertTTNNTensorAndValidate(op->v_out(), v);
}

void run(const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runSplitQueryKeyValueAndSplitHeadsOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::transformer
