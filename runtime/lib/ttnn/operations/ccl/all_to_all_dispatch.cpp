// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_dispatch.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllDispatchOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertIndices =
      tensorPool.getTTNNTensorAndValidate(op->expert_indices());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<uint32_t> axis =
      std::make_optional<uint32_t>(op->cluster_axis());

  auto [dispatched, metadata] =
      ::ttnn::all_to_all_dispatch(input, expertIndices, expertMapping,
                                  /*axis=*/axis,
                                  /*optional_output_tensors=*/std::nullopt,
                                  /*num_links=*/std::nullopt,
                                  /*topology=*/std::nullopt,
                                  /*memory_config=*/memoryConfig,
                                  /*subdevice_id=*/std::nullopt,
                                  /*output_concat_dim=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->dispatched(), dispatched);
  tensorPool.insertTTNNTensorAndValidate(op->metadata(), metadata);
}
} // namespace tt::runtime::ttnn::operations::ccl
