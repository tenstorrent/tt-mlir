// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_gather_minimal_matmul_async.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllGatherMinimalMatmulAsyncOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<float> scalar = std::nullopt;
  if (op->scalar()) {
    scalar = op->scalar().value();
  }

  std::optional<::ttnn::Tensor> addcmulInput1 = std::nullopt;
  if (op->addcmul_input1()) {
    addcmulInput1 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input1());
  }

  std::optional<::ttnn::Tensor> addcmulInput2 = std::nullopt;
  if (op->addcmul_input2()) {
    addcmulInput2 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input2());
  }

  // The all-gather is synchronized by two multi-device global semaphores.
  std::vector<::ttnn::GlobalSemaphore> multiDeviceSemaphore;
  for (const auto *semaphoreRef : *op->multi_device_semaphore()) {
    multiDeviceSemaphore.push_back(
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            semaphoreRef));
  }

  std::optional<::ttnn::GlobalSemaphore> barrierSemaphore = std::nullopt;
  if (op->barrier_semaphore()) {
    barrierSemaphore =
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            op->barrier_semaphore());
  }

  // Topology is a required metal parameter; default to Linear when unset.
  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*op->dtype());
  }

  std::optional<uint32_t> clusterAxis = std::nullopt;
  if (op->cluster_axis()) {
    clusterAxis = op->cluster_axis().value();
  }

  // The compiler doesn't set the matmul config or link topology yet, so we
  // build them here from the device grid. If we left the config empty, tt-metal
  // would run the matmul on the whole grid, which crashes for the two reasons
  // handled below.
  auto deviceGrid = context.getMeshDevice().compute_with_storage_grid_size();

  // (1) The all-gather puts its mux cores on the bottom row of the grid. If the
  //     matmul uses that row too, a corner core would run two data-movement
  //     kernels on the same NOC and hang. So we shrink the matmul grid by one
  //     row to leave the bottom row for the mux cores.
  uint32_t gridX = static_cast<uint32_t>(deviceGrid.x);
  uint32_t gridY = static_cast<uint32_t>(deviceGrid.y);
  // With 2 or fewer rows there's nothing left for the matmul after reserving
  // the mux row. Real multi-device grids are much taller, so this only trips on
  // degenerate/mock grids.
  LOG_ASSERT(gridY > 2,
             "all_gather_minimal_matmul_async requires a compute grid with "
             "more than 2 rows to reserve the all-gather mux row; got gridY=",
             gridY);
  gridY -= 1;

  // (2) The all-gather sends data along one axis of the grid (grid.x when
  //     force_transpose is set, otherwise grid.y). tt-metal splits that axis
  //     into num_links groups of num_workers_per_link cores each, and the split
  //     must come out exact. num_links only affects bandwidth, and the compiler
  //     doesn't set it, so we default to a single link that covers the whole
  //     axis (always a valid split). If the compiler ever sets num_links, use
  //     it as-is.
  uint32_t in0Axis = op->force_transpose() ? gridX : gridY;
  uint32_t numLinks;
  uint32_t numWorkersPerLink;
  if (op->num_links()) {
    numLinks = op->num_links().value();
    numWorkersPerLink = op->num_workers_per_link();
  } else {
    numWorkersPerLink = in0Axis;
    numLinks = 1;
  }
  // Every link uses 2 mux cores (one per direction). tt-metal checks these fit
  // on the grid and otherwise fails with a hard-to-read error. We add a simpler
  // check here. It's coarser than tt-metal's (which measures a different axis),
  // but it only trips on an unrealistically small grid, so that's fine.
  LOG_ASSERT(
      numLinks * 2 <= in0Axis,
      "all_gather_minimal_matmul_async needs num_links*2 (=", numLinks * 2,
      ") mux cores to fit on the in0 sender axis (=", in0Axis, ")");

  // These block sizes copy tt-metal's defaults: M/K/N blocks are always 8, and
  // fp32 accumulation (selected by leaving compute_kernel_config empty) uses
  // 2x2 subblocks. We clamp K_block_size ourselves: at most the K tiles per
  // device, and for Ring topology an exact divisor of it (Ring has no partial
  // block).
  // TODO(#0000): This copies tt-metal's fp32 default block sizes from a private
  //     helper we can't call, and is only right while compute_kernel_config is
  //     empty. The real fix is to set MinimalMatmulConfig in the compiler and
  //     pass it through.
  uint32_t kTilesPerDevice = static_cast<uint32_t>(input.padded_shape()[-1] /
                                                   ::tt::constants::TILE_WIDTH);
  uint32_t kBlockSize = std::min<uint32_t>(8, kTilesPerDevice);
  if (topology != ::ttnn::ccl::Topology::Linear) {
    while (kBlockSize > 1 && kTilesPerDevice % kBlockSize != 0) {
      --kBlockSize;
    }
  }
  kBlockSize = std::max<uint32_t>(1, kBlockSize);

  ::ttnn::experimental::prim::MinimalMatmulConfig matmulConfig;
  matmulConfig.M_block_size = 8;
  matmulConfig.K_block_size = kBlockSize;
  matmulConfig.N_block_size = 8;
  matmulConfig.subblock_h = 2;
  matmulConfig.subblock_w = 2;
  matmulConfig.compute_with_storage_grid_size =
      ::tt::tt_metal::CoreCoord{gridX, gridY};

  // `fused_activation`, `compute_kernel_config`, the persistent buffers and the
  // FSDP path are not modeled by the compiler yet; pass their tt-metal
  // defaults.
  std::vector<::ttnn::Tensor> outputs = ::ttnn::all_gather_minimal_matmul_async(
      input, weight, bias, scalar, addcmulInput1, addcmulInput2,
      /*fused_activation=*/std::nullopt, matmulConfig, multiDeviceSemaphore,
      topology, memoryConfig, dtype,
      /*compute_kernel_config=*/std::nullopt,
      /*persistent_output_buffer=*/std::nullopt, numLinks, clusterAxis,
      barrierSemaphore, op->force_transpose(), numWorkersPerLink,
      op->num_buffers_per_channel(), op->chunks(), op->dim());

  const auto *outputRefs = op->outputs();
  LOG_ASSERT(outputs.size() == outputRefs->size(),
             "all_gather_minimal_matmul_async produced ", outputs.size(),
             " outputs but the flatbuffer expects ", outputRefs->size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(outputRefs->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::ccl
