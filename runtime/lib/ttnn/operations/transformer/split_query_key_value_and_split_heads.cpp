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

  auto outputTuple = ::ttnn::transformer::split_query_key_value_and_split_heads(
      in, kvInputTensor, numHeads, numKVHeads, transposeKey,
      outputMemoryConfig);
  std::vector<::ttnn::Tensor> outputs = {std::get<0>(outputTuple),
                                         std::get<1>(outputTuple),
                                         std::get<2>(outputTuple)};

  LOG_ASSERT(
      op->outputs()->size() == outputs.size(),
      "Number of expected outputs does not match with generated outputs.");
  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}

void run(const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runSplitQueryKeyValueAndSplitHeadsOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
