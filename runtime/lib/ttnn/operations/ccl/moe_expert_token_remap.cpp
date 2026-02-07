// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/moe_expert_token_remap.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/data_movement/moe_expert_token_remap/moe_expert_token_remap.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MoeExpertTokenRemapOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &topkTensor =
      tensorPool.getTTNNTensorAndValidate(op->topk_tensor());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());
  const ::ttnn::Tensor &expertMetadata =
      tensorPool.getTTNNTensorAndValidate(op->expert_metadata());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  uint32_t reductionSize = op->reduction_size();

  auto results =
      ::ttnn::moe_expert_token_remap(topkTensor, expertMapping, expertMetadata,
                                     /*memory_config=*/memoryConfig,
                                     /*optional_output_tensor=*/std::nullopt,
                                     /*optional_reduced_tensor=*/std::nullopt,
                                     /*reduction_size=*/reductionSize);

  tensorPool.insertTTNNTensorAndValidate(op->mapping(), results[0]);
  tensorPool.insertTTNNTensorAndValidate(op->reduced(), results[1]);
}
} // namespace tt::runtime::ttnn::operations::ccl
