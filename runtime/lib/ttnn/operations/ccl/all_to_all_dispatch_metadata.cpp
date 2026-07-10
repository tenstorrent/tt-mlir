// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_dispatch_metadata.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.hpp"

#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::runtime::ttnn::operations::ccl {
namespace {
// Allocate a zero-initialized output buffer matching `tensorRef`'s shape/dtype.
// SPARSE_UNICAST only writes routed destination slots, so every other slot must
// already read as zero; reusing a stale buffer would leak routed data into
// newly-unrouted slots.
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

// Override the tensor's mesh topology in-place to axis-0 sharded along the
// ring, remaining axes replicated. Metadata-only (no device writes).
// `ttnn::zeros` produces a fully-replicated topology, but the dispatch kernel
// expects an axis-0 sharded view.
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
  // Axis 0: shard along tensor dim 0 (the ring dimension).
  placements.push_back(tt::tt_metal::distributed::MeshMapperConfig::Shard{0});
  // Remaining axes (e.g. data-parallel mesh_cols on a 2D mesh): replicated.
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

  // SPARSE_UNICAST token-leak fix (ported from the GPT-OSS e2e branch). The
  // library default SPARSE_MCAST_SHORTEST_PATH multi-path-writes into the same
  // destination slot, leaking tokens across the local batch on a (4,8) mesh ->
  // corrupt decode (1-layer decode PCC ~0.93). SPARSE_UNICAST point-to-point
  // routing with num_links=4 + pre-zeroed persistent outputs + a cross-device
  // semaphore fixes it.
  static constexpr uint32_t kDefaultNumLinks = 4;

  // Persistent pre-zeroed outputs. `ttnn::zeros` does a host->device write,
  // which is illegal inside a trace capture, so we must not call it fresh on
  // every invocation. The three outputs are handled differently based on how
  // the kernel writes them (ported from the GPT-OSS e2e branch):
  //
  //   * indices / scores: ALL-GATHER writes every slot, so initial contents
  //     are fully overwritten and the SAME buffer can be reused across calls.
  //     Route the zeros allocation through the implicit-tensor cache so the
  //     host write happens exactly once on the trace warmup func.call and the
  //     capture/replay calls read back the cached buffer. (These are L1
  //     height-sharded with non-tile-aligned shards, so the multiply trick
  //     below would hit a to_layout assert.)
  //
  //   * dispatched: SPARSE_UNICAST writes only routed slots, so it must be
  //     freshly zeroed every call — reusing it would leak stale routed data
  //     into newly-unrouted slots. Cache a zero TEMPLATE (host write once on
  //     warmup) and derive a fresh zeroed buffer each call via a device-side
  //     `ttnn::multiply` (dispatched is DRAM INTERLEAVED so multiply is legal
  //     and trace-safe). Mirrors the e2e IR `ttnn.full(0) -> ttnn.multiply`.
  //
  // All three still need axis-0 sharded topology (ttnn::zeros/multiply produce
  // replicated tensors by default but the kernel expects a sharded view).
  uintptr_t opKey = reinterpret_cast<uintptr_t>(op);
  ::ttnn::Tensor indicesOutput =
      context.getOrCreateImplicitTensor(opKey, /*subKey=*/1, [&]() {
        return createZeroOutput(op->indices(), context);
      });
  ::ttnn::Tensor scoresOutput =
      context.getOrCreateImplicitTensor(opKey, /*subKey=*/2, [&]() {
        return createZeroOutput(op->scores(), context);
      });
  ::ttnn::Tensor dispatchedTemplate =
      context.getOrCreateImplicitTensor(opKey, /*subKey=*/0, [&]() {
        return createZeroOutput(op->dispatched(), context);
      });
  ::ttnn::Tensor dispatchedOutput =
      ::ttnn::multiply(dispatchedTemplate, dispatchedTemplate);
  overrideAxis0ShardedTopology(dispatchedOutput, context);
  overrideAxis0ShardedTopology(indicesOutput, context);
  overrideAxis0ShardedTopology(scoresOutput, context);
  std::optional<std::array<::ttnn::Tensor, 3>> optionalOutputTensors =
      std::array<::ttnn::Tensor, 3>{dispatchedOutput, indicesOutput,
                                    scoresOutput};

  // Cross-device semaphore over all worker cores; SPARSE_UNICAST needs it for
  // the fabric point-to-point writes. Cached per-op via the ProgramContext.
  ::ttnn::MeshDevice *meshDevicePtr = context.getMeshDevicePtr().get();
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

  // With persistent pre-allocated outputs, the kernel does not need an explicit
  // drain_sync_tilizer_core (forcing one was observed to collapse the
  // dispatch).
  auto [dispatched, indices, scores] =
      ::ttnn::experimental::all_to_all_dispatch_metadata(
          input, expertIndices, expertScores, expertMapping,
          /*shared_expert_ids=*/std::nullopt,
          /*axis=*/axis,
          /*optional_output_tensors=*/optionalOutputTensors,
          /*num_links=*/std::make_optional<uint32_t>(kDefaultNumLinks),
          /*drain_sync_tilizer_core=*/std::nullopt,
          /*worker_mode=*/
          ::ttnn::operations::experimental::ccl::WorkerMode::DIRECT,
          /*dispatch_algorithm=*/
          ::ttnn::operations::experimental::ccl::DispatchAlgorithm::
              SPARSE_UNICAST,
          /*worker_core_range_set=*/std::nullopt,
          /*mux_core_range_set=*/std::nullopt,
          /*cross_device_semaphore=*/
          std::make_optional(crossDeviceSemaphore));

  tensorPool.insertTTNNTensorAndValidate(op->dispatched(), dispatched);
  tensorPool.insertTTNNTensorAndValidate(op->indices(), indices);
  tensorPool.insertTTNNTensorAndValidate(op->scores(), scores);
}
} // namespace tt::runtime::ttnn::operations::ccl
