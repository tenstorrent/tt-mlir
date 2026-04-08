// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/selective_reduce_combine.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::SelectiveReduceCombineOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &denseInputTensor =
      tensorPool.getTTNNTensorAndValidate(op->dense_input_tensor());
  const ::ttnn::Tensor &denseActivationsTensor =
      tensorPool.getTTNNTensorAndValidate(op->dense_activations_tensor());
  const ::ttnn::Tensor &denseTokenMapsTensor =
      tensorPool.getTTNNTensorAndValidate(op->dense_token_maps_tensor());
  const ::ttnn::Tensor &denseTokenCountsTensor =
      tensorPool.getTTNNTensorAndValidate(op->dense_token_counts_tensor());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<uint32_t> axis = std::make_optional<uint32_t>(op->axis());

  ::tt::tt_fabric::Topology topology =
      ::tt::runtime::common::toMetalTopology(op->topology());

  ::ttnn::Tensor output = ::ttnn::prim::selective_reduce_combine(
      denseInputTensor, denseActivationsTensor, denseTokenMapsTensor,
      denseTokenCountsTensor, op->hidden_size(), op->batch_size(),
      op->seq_size(), op->select_experts_k(), op->experts(), axis, topology,
      op->num_links(), op->num_token_parallel_cores(),
      op->num_data_parallel_cores(),
      /*worker_cores=*/{},
      /*mux_core_range_set=*/tt::tt_metal::CoreRangeSet{}, memoryConfig,
      /*optional_output_tensor=*/std::nullopt,
      /*optional_cross_device_semaphore=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
