// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/rotary_embedding_llama.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void
runRotaryEmbeddingLlama(const ::tt::target::ttnn::RotaryEmbeddingLlamaOp *op,
                        ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &cos_cache =
      tensorPool.getTTNNTensorAndValidate(op->cos_cache());
  const ::ttnn::Tensor &sin_cache =
      tensorPool.getTTNNTensorAndValidate(op->sin_cache());
  const ::ttnn::Tensor &tran_mat =
      tensorPool.getTTNNTensorAndValidate(op->tran_mat());
  bool is_decode_mode = op->is_decode_mode();
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }
  ::ttnn::Tensor out = ::ttnn::experimental::rotary_embedding_llama(
      in, cos_cache, sin_cache, tran_mat, is_decode_mode, outputMemoryConfig,
      computeConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::RotaryEmbeddingLlamaOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runRotaryEmbeddingLlama(op, tensorPool);
}

} // namespace tt::runtime::ttnn::operations::transformer
