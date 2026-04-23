// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/moe_gpt.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/ccl/moe_gpt/moe_gpt.hpp"

namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::MoeGptOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertIndices =
      tensorPool.getTTNNTensorAndValidate(op->expert_indices());
  const ::ttnn::Tensor &expertScores =
      tensorPool.getTTNNTensorAndValidate(op->expert_scores());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());
  const ::ttnn::Tensor &w0w1Tensor =
      tensorPool.getTTNNTensorAndValidate(op->w0_w1_tensor());
  const ::ttnn::Tensor &w2Tensor =
      tensorPool.getTTNNTensorAndValidate(op->w2_tensor());

  uint32_t outputHeightShardDim = op->output_height_shard_dim();
  uint32_t outputWidthShardDim = op->output_width_shard_dim();
  uint32_t hiddenSize = op->hidden_size();

  std::optional<uint32_t> clusterAxis = std::nullopt;
  if (op->cluster_axis().has_value()) {
    clusterAxis = op->cluster_axis().value();
  }

  auto results = ::ttnn::experimental::moe_gpt(
      inputTensor, expertIndices, expertScores, expertMapping, w0w1Tensor,
      w2Tensor, outputHeightShardDim, outputWidthShardDim, hiddenSize,
      clusterAxis);

  tensorPool.insertTTNNTensorAndValidate(op->token_counts(), results[0]);
  tensorPool.insertTTNNTensorAndValidate(op->activation_records(), results[1]);
  tensorPool.insertTTNNTensorAndValidate(op->token_indices(), results[2]);
  tensorPool.insertTTNNTensorAndValidate(op->tilize_out(), results[3]);
  tensorPool.insertTTNNTensorAndValidate(op->tilize_out_rm(), results[4]);
}
} // namespace tt::runtime::ttnn::operations::ccl
