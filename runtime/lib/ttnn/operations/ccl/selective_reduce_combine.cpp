// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/selective_reduce_combine.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"

namespace tt::runtime::ttnn::operations::ccl {

// GPT-OSS device-specific params not modeled in the IR.
static constexpr uint32_t kDefaultClusterAxis = 0;
static constexpr uint32_t kDefaultNumLinks = 4;
// Combine worker grid: 4 token-parallel (rows) x 3 data-parallel (cols) = 12
// cores. Batch is DP-sharded (32 tokens/device), so total ring tokens = 128 and
// the combine circular buffer fits the L1 bank on 12 cores.
static constexpr uint32_t kCombineTokenParallelCoreDim =
    4; // rows (= moe_gpt output_height_shard_dim)
static constexpr uint32_t kCombineDataParallelCoreDim =
    3; // cols (= moe_gpt output_width_shard_dim)

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

  // Combine worker cores: a 4(token-parallel rows) x 3(data-parallel cols)
  // rectangle at (1,0)-(3,3), with mux 2 columns to the right at (4,0)-(5,3).
  std::vector<::ttnn::CoreCoord> workerCores;
  for (uint32_t y = 0; y < kCombineTokenParallelCoreDim; ++y) {
    for (uint32_t x = 1; x <= kCombineDataParallelCoreDim; ++x) {
      workerCores.emplace_back(x, y);
    }
  }
  tt::tt_metal::CoreRangeSet muxCores({tt::tt_metal::CoreRange(
      {kCombineDataParallelCoreDim + 1, 0},
      {kCombineDataParallelCoreDim + 2, kCombineTokenParallelCoreDim - 1})});

  // The combine kernel performs sparse writes into the output buffer; the
  // compiler (TTIRToTTNN) provides a zero-initialized tensor via
  // `optional_output_tensor` (emitted as `ttnn.full(0)` prior to this op, made
  // per-combine-distinct so const-eval caches one buffer per layer rather than
  // sharing a single buffer across all combines). Without this pre-zeroing,
  // uninitialized DRAM slots would leak -inf / NaN into the downstream
  // score-weighted sum and all_reduce.
  std::optional<::ttnn::Tensor> optionalOutputTensor;
  if (op->optional_output_tensor()) {
    optionalOutputTensor =
        tensorPool.getTTNNTensorAndValidate(op->optional_output_tensor());
  }

  // Cross-device semaphore for the both-path combine. Providing BOTH the
  // pre-zeroed output AND this semaphore enables the optimized fabric-EDM path
  // that performs the cross-device combine writes directly; without it the
  // program factory falls back to the init-semaphore path, which leaves
  // destination slots on remote ring devices untouched (residual-only outputs)
  // and drops decode PCC.
  ::ttnn::MeshDevice *meshDevicePtr = context.getMeshDevicePtr().get();
  // Cache the semaphore via the ProgramContext: create_global_semaphore does a
  // host->device L1 write, which is illegal during trace capture. Caching moves
  // it to the trace warmup func.call and reuses it on capture/replay.
  ::ttnn::GlobalSemaphore crossDeviceSemaphore =
      context.getOrCreateImplicitGlobalSemaphore(
          reinterpret_cast<uintptr_t>(op), [&]() {
            auto computeGrid = meshDevicePtr->compute_with_storage_grid_size();
            ::tt::tt_metal::CoreRangeSet semaphoreCores(
                ::tt::tt_metal::CoreRange(
                    ::tt::tt_metal::CoreCoord(0, 0),
                    ::tt::tt_metal::CoreCoord(computeGrid.x - 1,
                                              computeGrid.y - 1)));
            return ::ttnn::global_semaphore::create_global_semaphore(
                meshDevicePtr, semaphoreCores, /*initial_value=*/0);
          });

  // moe_gpt output[4] is a ROW_MAJOR alias of the HEIGHT_SHARDED activation
  // buffer. The TTNN graph may insert a to_layout(TILE) before this op to
  // satisfy generic layout planning, but the selective_reduce_combine kernel
  // aliases the dense input buffer and uses row-major page/byte addressing.
  // Feed it a ROW_MAJOR tensor here so the physical layout matches the kernel
  // contract.
  ::ttnn::Tensor denseInputForKernel =
      denseInputTensor.layout() == ::ttnn::Layout::ROW_MAJOR
          ? denseInputTensor
          : ::ttnn::to_layout(denseInputTensor, ::ttnn::Layout::ROW_MAJOR);

  ::ttnn::Tensor output = ::ttnn::prim::selective_reduce_combine(
      denseInputForKernel, denseActivationsTensor, denseTokenMapsTensor,
      denseTokenCountsTensor, op->hidden_size(), op->batch_size(),
      op->seq_size(), op->select_experts_k(),
      /*axis=*/kDefaultClusterAxis,
      // Linear (not Ring): the combine performs per-ring-position cross-device
      // scatter writes; under a Ring/FABRIC_1D_RING wrap the last->first wrap
      // corrupts those writes and decode output diverges per batch (the auto
      // FABRIC_1D_RING fabric is overridden to FABRIC_1D in tt-xla for the same
      // reason). Linear maps the writes 1:1 to devices.
      /*topology=*/::tt::tt_fabric::Topology::Linear,
      /*num_links=*/kDefaultNumLinks,
      /*num_token_parallel_cores=*/kCombineTokenParallelCoreDim,
      /*num_data_parallel_cores=*/kCombineDataParallelCoreDim,
      /*worker_cores=*/workerCores,
      /*mux_core_range_set=*/muxCores,
      /*output_memory_config=*/std::nullopt,
      /*optional_output_tensor=*/optionalOutputTensor,
      /*optional_cross_device_semaphore=*/
      std::make_optional(crossDeviceSemaphore));

  // The selective_reduce_combine kernel emits its result in ROW_MAJOR layout.
  // The compiled IR / flatbuffer may expect a different layout for op->out()
  // (the GPT-OSS decode graph assigns TILE), and forcing the IR result layout
  // to ROW_MAJOR sends the layout optimizer into a non-terminating
  // reconciliation. Reconcile here instead: convert the kernel output to the
  // layout op->out() expects so insertTTNNTensorAndValidate's layout check
  // (inferLayoutFromTileShape) passes and downstream consumers see the layout
  // the compiler planned for.
  ::ttnn::Layout expectedLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
  if (output.layout() != expectedLayout) {
    output = ::ttnn::to_layout(output, expectedLayout);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
