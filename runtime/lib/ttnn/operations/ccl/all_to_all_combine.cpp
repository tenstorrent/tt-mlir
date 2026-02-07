// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_combine.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllCombineOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());
  const ::ttnn::Tensor &expertMetadata =
      tensorPool.getTTNNTensorAndValidate(op->expert_metadata());

  std::optional<uint32_t> axis =
      std::make_optional<uint32_t>(op->cluster_axis());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  ::ttnn::Tensor output = ::ttnn::all_to_all_combine(
      input, expertMapping, expertMetadata,
      /*locally_reduced=*/false,
      /*num_links=*/std::nullopt,
      /*topology=*/std::nullopt,
      /*memory_config=*/memoryConfig,
      /*axis=*/axis,
      /*output_shard_dim=*/std::nullopt,
      /*subdevice_id=*/std::nullopt,
      /*optional_output_tensor=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
