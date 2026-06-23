// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/selective_reduce_combine.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/global_semaphore.hpp"

#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::runtime::ttnn::operations::ccl {

// GPT-OSS device-specific params not modeled in the IR.
static constexpr uint32_t kDefaultClusterAxis = 0;
static constexpr uint32_t kDefaultNumLinks = 4;
// Combine worker grid: 4 token-parallel (rows) x 3 data-parallel (cols) = 12
// cores, matching the e2e branch and the tt-metal GPT-OSS demo. The batch is
// DP-sharded (32 tokens/device) by the dispatch/moe_gpt/combine sharding rules
// (B factor = kPassThrough), so total ring tokens = 128 and the combine
// circular buffer fits the L1 bank on 12 cores. (An earlier replicated-batch
// layout forced 512 tokens and needed 40 cores, which broke the combine.)
static constexpr uint32_t kCombineTokenParallelCoreDim = 4; // rows
static constexpr uint32_t kCombineDataParallelCoreDim = 3;  // cols

namespace {
// Allocate a zero-initialized output buffer matching `tensorRef`'s shape/dtype.
// The combine kernel writes into this buffer in place via the fabric EDM
// cross-device path; pre-zeroing guarantees unwritten (non-source) slots read
// as zero instead of leaking stale data. (Ported from the GPT-OSS e2e branch /
// d9679a709 "Fix for pcc issues".)
::ttnn::Tensor createZeroOutput(const ::tt::target::ttnn::TensorRef *tensorRef,
                                ProgramContext &context) {
  ::ttnn::Shape shape = ::tt::runtime::ttnn::operations::utils::toTTNNShape(
      *tensorRef->desc()->shape());
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(tensorRef);
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(tensorRef);
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(tensorRef));

  ::ttnn::MeshDevice *meshDevicePtr = context.getMeshDevicePtr().get();
  ::tt::runtime::ttnn::OptionalMeshDeviceRef meshDevice =
      std::ref(*meshDevicePtr);
  return ::ttnn::zeros(shape, dtype, layout, meshDevice, memoryConfig);
}

// Override the tensor's mesh topology in-place to axis-0 sharded along the ring,
// remaining axes replicated. Metadata-only (no device writes). `ttnn::zeros`
// produces a fully-replicated topology, but the combine kernel expects an
// axis-0 sharded view (it writes per-ring-position shards). (Ported from the
// GPT-OSS e2e branch.)
void overrideAxis0ShardedTopology(::ttnn::Tensor &tensor,
                                  ProgramContext &context) {
  ::ttnn::MeshDevice *meshDevicePtr = context.getMeshDevicePtr().get();
  const auto &meshShape = meshDevicePtr->shape();
  if (meshShape.dims() < 1) {
    return;
  }
  ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>
      placements;
  placements.reserve(meshShape.dims());
  placements.push_back(tt::tt_metal::distributed::MeshMapperConfig::Shard{0});
  for (size_t i = 1; i < meshShape.dims(); ++i) {
    placements.push_back(
        tt::tt_metal::distributed::MeshMapperConfig::Replicate{});
  }

  std::vector<tt::tt_metal::distributed::MeshCoordinate> coords;
  coords.reserve(meshShape.mesh_size());
  for (const auto &coord :
       tt::tt_metal::distributed::MeshCoordinateRange(meshShape)) {
    coords.push_back(coord);
  }

  tt::tt_metal::TensorTopology topology(meshShape, std::move(placements),
                                        std::move(coords));
  tensor.update_tensor_topology(std::move(topology));
}
} // namespace

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
  // rectangle at (1,0)-(3,3), with mux 2 columns to the right at (4,0)-(5,3)
  // (matches the e2e branch / GPT-OSS demo layout).
  std::vector<::ttnn::CoreCoord> workerCores;
  for (uint32_t y = 0; y < kCombineTokenParallelCoreDim; ++y) {
    for (uint32_t x = 1; x <= kCombineDataParallelCoreDim; ++x) {
      workerCores.emplace_back(x, y);
    }
  }
  tt::tt_metal::CoreRangeSet muxCores({tt::tt_metal::CoreRange(
      {kCombineDataParallelCoreDim + 1, 0},
      {kCombineDataParallelCoreDim + 2, kCombineTokenParallelCoreDim - 1})});

  // Both-path combine (matches e2e d9679a709 "Fix for pcc issues"). Our
  // reconstruction's SelectiveReduceCombineOp has no optional_output_tensor
  // flatbuffer operand, so we allocate the pre-zeroed output here instead of
  // reading op->optional_output_tensor(). Providing BOTH a pre-zeroed output
  // tensor AND a cross-device semaphore enables the optimized fabric-EDM path
  // that performs the cross-device combine writes directly; without them the
  // program factory falls back to the init-semaphore path, which leaves
  // destination slots on remote ring devices untouched (residual-only outputs)
  // and drops decode PCC (~0.947 vs ~0.969).
  ::ttnn::Tensor zeroOutput = createZeroOutput(op->out(), context);
  overrideAxis0ShardedTopology(zeroOutput, context);

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
      op->seq_size(), op->select_experts_k(),
      /*axis=*/kDefaultClusterAxis,
      /*topology=*/::tt::tt_fabric::Topology::Linear,
      /*num_links=*/kDefaultNumLinks,
      /*num_token_parallel_cores=*/kCombineTokenParallelCoreDim,
      /*num_data_parallel_cores=*/kCombineDataParallelCoreDim,
      /*worker_cores=*/workerCores,
      /*mux_core_range_set=*/muxCores,
      /*output_memory_config=*/std::nullopt,
      /*optional_output_tensor=*/std::make_optional(zeroOutput),
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
