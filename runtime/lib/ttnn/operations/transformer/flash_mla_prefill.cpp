// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/flash_mla_prefill.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void
runFlashMlaPrefillOp(const ::tt::target::ttnn::FlashMlaPrefillOp *op,
                     ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());

  const std::optional<::ttnn::Tensor> value =
      op->value()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->value()))
          : std::nullopt;

  const std::optional<::ttnn::Tensor> attentionMask =
      op->attention_mask()
          ? std::make_optional(
                tensorPool.getTTNNTensorAndValidate(op->attention_mask()))
          : std::nullopt;

  uint32_t headDimV = op->head_dim_v();
  bool isCausal = op->is_causal();
  std::optional<float> scale = op->scale();

  ::ttnn::Tensor out = ::ttnn::transformer::flash_mla_prefill(
      query, key, headDimV, value, attentionMask, isCausal, scale,
      outputMemoryConfig,
      /*program_config=*/std::nullopt,
      /*compute_kernel_config=*/std::nullopt);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::FlashMlaPrefillOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runFlashMlaPrefillOp(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
