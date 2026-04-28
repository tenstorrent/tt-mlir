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
// Allocate a zero-initialized device tensor matching the shape/dtype/layout/
// memory_config of `tensorRef`. Required so that SPARSE_UNICAST dispatch
// leaves unrouted destination slots as zero instead of stale/garbage data.
//
// IMPORTANT: the tt-metal GPT-OSS reference
// (models/demos/gpt_oss/tt/experts_throughput/weights.py) builds these
// pre-allocated output buffers with
//   ShardTensor2dMesh(dims=(0, None), mesh_shape=(ring_devices, mesh_cols))
// so the kernel sees the dispatched outputs as axis-0 sharded over the ring
// (global dim 0 == ring_devices, per-device dim 0 == 1), replicated along
// the orthogonal mesh axis. ttnn::zeros (mesh overload) produces a fully
// replicated tensor instead, and the resulting replicated-tensor metadata
// makes the SPARSE_UNICAST fabric writes mis-target the output slot:
// every device ends up writing to slot 0..local_batch-1, which collapses the
// cross-device dispatch (all devices downstream of the source produce
// identical wrong outputs).
//
// Used as a fallback only when the IR did not provide
// `optional_*_output_tensor` operands. The preferred path is to pre-zero
// the buffers in the IR via `ttnn.full -> ttnn.multiply` so the runtime
// never has to call `ttnn::zeros` (which writes initial values to device
// memory and is therefore incompatible with trace capture).
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

// Override the tensor's mesh topology in-place to match what the dispatch
// kernel expects (see createZeroOutput comment): axis-0 sharded along the
// ring, axis-1 replicated. This is metadata-only — no device writes — so
// it is safe to invoke during trace capture. Required regardless of how the
// underlying storage was allocated (`ttnn::zeros` or
// `ttnn.full + ttnn.multiply` from the IR), because all of those default to
// a fully-replicated topology.
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

  // We always pass persistent pre-allocated output tensors below, so the
  // kernel does not need an explicit drain_sync_tilizer_core. Matches the
  // tt-metal GPT-OSS reference
  // (models/demos/gpt_oss/tt/experts_throughput/fused_decode.py) which does
  // not pass drain_sync_tilizer_core at all. Forcing (0,0) here was observed
  // to collapse the cross-device dispatch: only the source device wrote its
  // portion of the dispatched tensor, leaving devices 1..N-1 with the zero-
  // initialized output and producing identical wrong outputs on them.
  std::optional<tt::tt_metal::CoreCoord> drainCore = std::nullopt;

  // Match tt-metal GPT-OSS reference
  // (models/demos/gpt_oss/tt/experts_throughput/fused_decode.py): use
  // SPARSE_UNICAST point-to-point routing with num_links=4. The library
  // default (SPARSE_MCAST_SHORTEST_PATH) causes multi-path writes into the
  // same destination slot, leaking tokens across the local batch on a (4,8)
  // mesh.
  static constexpr uint32_t kDefaultNumLinks = 4;

  // Pre-allocate zero-initialized output tensors for SPARSE_UNICAST.
  // Without this, unrouted slots on non-source devices retain uninitialized
  // DRAM contents, which manifests as all-zero or garbage MoE outputs on
  // devices 1..N-1 of each row in a multi-device mesh.
  //
  // Materialization strategy (split because the dispatch kernel writes the
  // three buffers very differently):
  //
  //   * `dispatched`: SPARSE_UNICAST writes — only routed destination slots
  //     are touched, every other slot must remain zero. The buffer must be
  //     freshly zeroed on every call; reusing it across calls would leak
  //     stale routed data into newly-unrouted slots and corrupt downstream
  //     MoE compute. Compiled into the IR by
  //     `TTIRToTTNN::AllToAllDispatchMetadataOpConversionPattern` as
  //     `ttnn.full(0) -> ttnn.multiply(zero, zero)` (const-eval'd outside
  //     `@main`), with a per-call duplicator multiplier spliced in by
  //     `TTNNMoEOpsWorkaround::AllToAllDispatchMetadataOptionalOutputDup-
  //     licatePattern` so the kernel sees a fresh DRAM tensor each call.
  //
  //   * `indices`/`scores`: ALL-GATHER writes — every slot is written with
  //     the corresponding source device's value, so initial buffer contents
  //     are completely overwritten and reusing the same buffer across calls
  //     is safe. The IR-level multiply trick does not work for these
  //     L1 height-sharded buffers (shard shape is not tile-aligned, so the
  //     `ttnn::multiply` runtime hits a `to_layout` assert). Instead, route
  //     through `ProgramContext::getOrCreateImplicitTensor` so the host
  //     write (`ttnn::zeros`) happens exactly once during the trace warmup
  //     pass and the cached buffer is reused inside the captured trace.
  //
  // All three buffers still need their mesh topology overridden to axis-0
  // sharded (see `overrideAxis0ShardedTopology`), since both `ttnn::zeros`
  // and `ttnn::full` produce replicated tensors by default but the kernel
  // expects a sharded view. `update_tensor_topology` is metadata-only and
  // trace-safe.
  uintptr_t opKey = reinterpret_cast<uintptr_t>(op);
  ::ttnn::Tensor dispatchedOutput =
      op->optional_dispatched_output_tensor()
          ? tensorPool.getTTNNTensorAndValidate(
                op->optional_dispatched_output_tensor())
          : createZeroOutput(op->dispatched(), context);
  ::ttnn::Tensor indicesOutput =
      op->optional_indices_output_tensor()
          ? tensorPool.getTTNNTensorAndValidate(
                op->optional_indices_output_tensor())
          : context.getOrCreateImplicitTensor(opKey, /*subKey=*/1, [&]() {
              return createZeroOutput(op->indices(), context);
            });
  ::ttnn::Tensor scoresOutput =
      op->optional_scores_output_tensor()
          ? tensorPool.getTTNNTensorAndValidate(
                op->optional_scores_output_tensor())
          : context.getOrCreateImplicitTensor(opKey, /*subKey=*/2, [&]() {
              return createZeroOutput(op->scores(), context);
            });

  overrideAxis0ShardedTopology(dispatchedOutput, context);
  overrideAxis0ShardedTopology(indicesOutput, context);
  overrideAxis0ShardedTopology(scoresOutput, context);

  std::optional<std::array<::ttnn::Tensor, 3>> optionalOutputTensors =
      std::array<::ttnn::Tensor, 3>{dispatchedOutput, indicesOutput,
                                    scoresOutput};

  // Create cross-device semaphore spanning all worker cores of the mesh
  // device. SPARSE_UNICAST requires this for cross-device fabric writes; when
  // a semaphore is provided together with persistent outputs, the op runs in
  // the optimized "semaphore-free" mode and the fabric EDM actually performs
  // the point-to-point writes to the remote devices. Matches tt-metal
  // GPT-OSS reference (experts_throughput/weights.py):
  //   compute_grid = mesh_device.compute_with_storage_grid_size()
  //   all_worker_cores =
  //       CoreRangeSet({CoreRange((0,0), (grid.x-1, grid.y-1))})
  //   dispatch_semaphore =
  //       ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)
  //
  // `create_global_semaphore` writes the initial value to all devices via
  // `EnqueueWriteMeshBuffer`, which trace capture rejects. Route through
  // `ProgramContext::getOrCreateImplicitGlobalSemaphore` so the semaphore is
  // created exactly once (during the trace warmup pass before
  // `begin_trace_capture`) and reused for the captured trace and all
  // subsequent decode steps.
  ::ttnn::MeshDevice *meshDevicePtr = context.getMeshDevicePtr().get();
  ::ttnn::GlobalSemaphore crossDeviceSemaphore =
      context.getOrCreateImplicitGlobalSemaphore(
          reinterpret_cast<uintptr_t>(op), [&]() {
            auto computeGrid = meshDevicePtr->compute_with_storage_grid_size();
            ::tt::tt_metal::CoreRangeSet semaphoreCores(
                ::tt::tt_metal::CoreRange(::tt::tt_metal::CoreCoord(0, 0),
                                          ::tt::tt_metal::CoreCoord(
                                              computeGrid.x - 1,
                                              computeGrid.y - 1)));
            return ::ttnn::global_semaphore::create_global_semaphore(
                meshDevicePtr, semaphoreCores, /*initial_value=*/0);
          });

  auto [dispatched, indices, scores] =
      ::ttnn::experimental::all_to_all_dispatch_metadata(
          input, expertIndices, expertScores, expertMapping,
          /*shared_expert_ids=*/std::nullopt,
          /*axis=*/axis,
          /*optional_output_tensors=*/optionalOutputTensors,
          /*num_links=*/std::make_optional<uint32_t>(kDefaultNumLinks),
          /*drain_sync_tilizer_core=*/drainCore,
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
