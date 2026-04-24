// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/selective_reduce_combine.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/global_semaphore.hpp"

namespace tt::runtime::ttnn::operations::ccl {

// GPT-OSS defaults for device-specific params not modeled in the IR.
// Values taken from tt-metal test:
// tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py
// (run_test_dispatch_compute_combine, mesh=(4,8), GPT-OSS config)
static constexpr uint32_t kDefaultClusterAxis = 0;
static constexpr uint32_t kDefaultNumLinks = 4;
static constexpr uint32_t kDefaultTokenParallelCoreDim = 4;
static constexpr uint32_t kDefaultDataParallelCoreDim = 3;

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

  // GPT-OSS worker cores: 3x4 rectangle at (1,0)-(3,3).
  std::vector<::ttnn::CoreCoord> workerCores;
  for (uint32_t y = 0; y < kDefaultTokenParallelCoreDim; ++y) {
    for (uint32_t x = 1; x <= kDefaultDataParallelCoreDim; ++x) {
      workerCores.emplace_back(x, y);
    }
  }

  // GPT-OSS mux cores: 2x4 rectangle at (4,0)-(5,3).
  tt::tt_metal::CoreRangeSet muxCores(
      {tt::tt_metal::CoreRange({4, 0}, {5, 3})});

  // The combine kernel performs sparse writes into the output buffer; the
  // compiler (TTIRToTTNN) is responsible for providing a zero-initialized
  // tensor via `optional_output_tensor` (emitted as `ttnn.full(0)` prior to
  // this op). Without that pre-zeroing, uninitialized DRAM slots would leak
  // -inf / NaN into the downstream score-weighted sum and all_reduce.
  std::optional<::ttnn::Tensor> optionalOutputTensor;
  if (op->optional_output_tensor()) {
    optionalOutputTensor =
        tensorPool.getTTNNTensorAndValidate(op->optional_output_tensor());
  }

  // Match tt-metal GPT-OSS reference (experts_throughput/weights.py):
  //   combine_semaphore =
  //       ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)
  // Providing both `optional_output_tensor` and
  // `optional_cross_device_semaphore` enables the optimized "semaphore-free"
  // path where the fabric EDM performs the cross-device combine writes
  // directly. Without the semaphore, the program factory falls back to the
  // init-semaphore path which may leave destination slots on remote devices
  // untouched, manifesting as identical (residual-only) outputs on
  // non-source devices of the cluster row.
  ::ttnn::MeshDevice *meshDevicePtr = context.getMeshDevicePtr().get();
  auto computeGrid = meshDevicePtr->compute_with_storage_grid_size();
  ::tt::tt_metal::CoreRangeSet semaphoreCores(::tt::tt_metal::CoreRange(
      ::tt::tt_metal::CoreCoord(0, 0),
      ::tt::tt_metal::CoreCoord(computeGrid.x - 1, computeGrid.y - 1)));
  ::ttnn::GlobalSemaphore crossDeviceSemaphore =
      ::ttnn::global_semaphore::create_global_semaphore(
          meshDevicePtr, semaphoreCores, /*initial_value=*/0);

  ::ttnn::Tensor output = ::ttnn::prim::selective_reduce_combine(
      denseInputTensor, denseActivationsTensor, denseTokenMapsTensor,
      denseTokenCountsTensor, op->hidden_size(), op->batch_size(),
      op->seq_size(), op->select_experts_k(), op->experts(),
      /*axis=*/std::make_optional<uint32_t>(kDefaultClusterAxis),
      /*topology=*/::tt::tt_fabric::Topology::Ring,
      /*num_links=*/kDefaultNumLinks,
      /*num_token_parallel_cores=*/kDefaultTokenParallelCoreDim,
      /*num_data_parallel_cores=*/kDefaultDataParallelCoreDim,
      /*worker_cores=*/workerCores,
      /*mux_core_range_set=*/muxCores,
      /*output_memory_config=*/std::nullopt,
      /*optional_output_tensor=*/optionalOutputTensor,
      /*optional_cross_device_semaphore=*/
      std::make_optional(crossDeviceSemaphore));

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
