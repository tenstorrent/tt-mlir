// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/rotary_embedding.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
static void runRotaryEmbedding(const ::tt::target::ttnn::RotaryEmbeddingOp *op,
                               ProgramTensorPool &tensorPool) {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &cosCache =
      tensorPool.getTTNNTensorAndValidate(op->cos_cache());
  const ::ttnn::Tensor &sinCache =
      tensorPool.getTTNNTensorAndValidate(op->sin_cache());
  std::optional<uint32_t> tokenIndex = op->token_index();
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig(
      ::ttnn::WormholeComputeKernelConfig{});

  // https://github.com/tenstorrent/tt-mlir/issues/5790
  std::get<::ttnn::WormholeComputeKernelConfig>(*computeConfig).math_fidelity =
      MathFidelity::HiFi4;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::Tensor out = ::ttnn::experimental::rotary_embedding(
      in, cosCache, sinCache, tokenIndex, outputMemoryConfig, computeConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::RotaryEmbeddingOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runRotaryEmbedding(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::transformer
