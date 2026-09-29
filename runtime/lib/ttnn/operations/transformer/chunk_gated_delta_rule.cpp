// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/chunk_gated_delta_rule.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::ChunkGatedDeltaRuleOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  auto tensor =
      [&](const ::tt::target::ttnn::TensorRef *ref) -> const ::ttnn::Tensor & {
    return tensorPool.getTTNNTensorAndValidate(ref);
  };
  auto optionalTensor = [&](const ::tt::target::ttnn::TensorRef *ref)
      -> std::optional<::ttnn::Tensor> {
    return ref ? std::make_optional(tensor(ref)) : std::nullopt;
  };

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  auto [output, finalState] = ::ttnn::transformer::chunk_gated_delta_rule(
      tensor(op->query()), tensor(op->key()), tensor(op->value()),
      tensor(op->g()), tensor(op->beta()), op->scale(),
      optionalTensor(op->initial_state()), op->output_final_state(),
      op->chunk_size(), op->use_qk_l2norm(), op->output_head_major(),
      /*use_mcast=*/true, memoryConfig, computeConfig,
      optionalTensor(op->eye()), optionalTensor(op->tril()),
      optionalTensor(op->ones()), optionalTensor(op->masks()));

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
  if (op->final_state()) {
    TT_FATAL(finalState.has_value(),
             "chunk_gated_delta_rule did not return requested final state");
    tensorPool.insertTTNNTensorAndValidate(op->final_state(), *finalState);
  }
}

} // namespace tt::runtime::ttnn::operations::transformer
