// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_dispatch_metadata.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllDispatchMetadataOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertIndices =
      tensorPool.getTTNNTensorAndValidate(op->expert_indices());
  const ::ttnn::Tensor &expertScores =
      tensorPool.getTTNNTensorAndValidate(op->expert_scores());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());

  std::optional<uint32_t> axis =
      std::make_optional<uint32_t>(op->cluster_axis());

  // This op runs in persistent mode only: the three output buffers and the
  // cross-device semaphore are bound in the prelude. Passing them makes
  // tt-metal set SKIP_INIT_SEMAPHORE; the drain core is left nullopt (the
  // kernel derives it from the shard spec).
  LOG_ASSERT(op->dispatched_buffer() && op->indices_buffer() &&
                 op->scores_buffer() && op->cross_device_semaphore(),
             "all_to_all_dispatch_metadata requires the persistent output "
             "buffers and cross-device semaphore to be bound");

  std::array<::ttnn::Tensor, 3> optionalOutputs{
      tensorPool.getTTNNTensorAndValidate(op->dispatched_buffer()),
      tensorPool.getTTNNTensorAndValidate(op->indices_buffer()),
      tensorPool.getTTNNTensorAndValidate(op->scores_buffer())};

  ::ttnn::GlobalSemaphore crossDeviceSemaphore =
      context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
          op->cross_device_semaphore());

  auto [dispatched, indices, scores] =
      ::ttnn::experimental::all_to_all_dispatch_metadata(
          input, expertIndices, expertScores, expertMapping,
          /*shared_expert_ids=*/std::nullopt,
          /*axis=*/axis,
          /*optional_output_tensors=*/optionalOutputs,
          /*num_links=*/std::nullopt,
          /*drain_sync_tilizer_core=*/std::nullopt,
          /*worker_mode=*/
          ::ttnn::operations::experimental::ccl::WorkerMode::DIRECT,
          // SPARSE_UNICAST routes per-target via the correct get_route(),
          // matching tt-metal's gpt_oss reference. The op default
          // SPARSE_MCAST_SHORTEST_PATH has a tt-metal-documented cluster_axis=0
          // bug (it computes ring hop distances from the global linearized
          // device id instead of the intra-ring column position -> wrong
          // target). No MLIR attr exposes this yet, so it is hardcoded here.
          /*dispatch_algorithm=*/
          ::ttnn::operations::experimental::ccl::DispatchAlgorithm::
              SPARSE_UNICAST,
          /*worker_core_range_set=*/std::nullopt,
          /*mux_core_range_set=*/std::nullopt,
          /*cross_device_semaphore=*/crossDeviceSemaphore);

  tensorPool.insertTTNNTensorAndValidate(op->dispatched(), dispatched);
  tensorPool.insertTTNNTensorAndValidate(op->indices(), indices);
  tensorPool.insertTTNNTensorAndValidate(op->scores(), scores);
}
} // namespace tt::runtime::ttnn::operations::ccl
