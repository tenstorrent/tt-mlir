// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/minimal_matmul_strided_reduce_scatter_async.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/minimal_matmul_strided_reduce_scatter_async.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

#include <tt-metalium/constants.hpp>

#include <algorithm>

namespace tt::runtime::ttnn::operations::ccl {

namespace {
// Largest divisor of `value` that is <= `cap` (and >= 1). Used to pick
// core-grid extents that divide the tile counts evenly, as the kernels
// require.
uint32_t largestDivisorAtMost(uint32_t value, uint32_t cap) {
  for (uint32_t d = std::min(value, cap); d >= 1; --d) {
    if (value % d == 0) {
      return d;
    }
  }
  return 1;
}
} // namespace

void run(const ::tt::target::ttnn::MinimalMatmulStridedReduceScatterAsyncOp *op,
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

  std::optional<::ttnn::Tensor> addcmulInput1 = std::nullopt;
  if (op->addcmul_input1()) {
    addcmulInput1 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input1());
  }

  std::optional<::ttnn::Tensor> addcmulInput2 = std::nullopt;
  if (op->addcmul_input2()) {
    addcmulInput2 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input2());
  }

  std::optional<float> scalar = std::nullopt;
  if (op->scalar()) {
    scalar = op->scalar().value();
  }

  // The async reduce-scatter is synchronized by the multi-device global
  // semaphores plus an optional barrier semaphore.
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

  // Topology is a required metal parameter; default to Ring when unset.
  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Ring;
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

  uint32_t numLinks = op->num_links() ? op->num_links().value() : 1;

  std::optional<uint32_t> clusterAxis = std::nullopt;
  if (op->cluster_axis()) {
    clusterAxis = op->cluster_axis().value();
  }

  // compute_kernel_config is a required tt-metal parameter. Build one from the
  // op attribute when present, otherwise fall back to the device-default.
  ::ttnn::DeviceComputeKernelConfig computeKernelConfig =
      op->compute_config()
          ? utils::createDeviceComputeKernelConfig(op->compute_config())
          : ::ttnn::init_device_compute_kernel_config(
                context.getMeshDevice().arch(), /*device_kernel_config=*/
                std::nullopt);

  // tt-metal requires a MinimalMatmulConfig and the reduce-scatter cores must
  // sit below the matmul grid. The compiler does not model this yet, so derive
  // a functional (not perf-tuned) config from the problem shape.
  // TODO(#0000): model MinimalMatmulConfig in the compiler and thread it here.
  auto deviceGrid = context.getMeshDevice().compute_with_storage_grid_size();
  uint32_t gridX = static_cast<uint32_t>(deviceGrid.x);
  uint32_t gridY = static_cast<uint32_t>(deviceGrid.y);

  // Tile counts. input is [1, 1, M, K/ring], weight is [1, 1, K/ring, N];
  // the matmul output is [1, 1, M, N].
  uint32_t mTiles = static_cast<uint32_t>(input.padded_shape()[-2] /
                                          ::tt::constants::TILE_HEIGHT);
  uint32_t kTiles = static_cast<uint32_t>(input.padded_shape()[-1] /
                                          ::tt::constants::TILE_WIDTH);
  uint32_t nTiles = static_cast<uint32_t>(weight.padded_shape()[-1] /
                                          ::tt::constants::TILE_WIDTH);

  // Fail cleanly on configurations tt-metal's fused kernel cannot run, rather
  // than dispatching them and hanging the mesh (a device deadlock leaves the
  // fabric wedged and needs a reset). Two known-bad cases:
  //   1. A degenerate reduce-scatter ring: the kernel deadlocks on a 2-device
  //      ring (nightly coverage uses an 8-device ring); it needs > 2 devices.
  //   2. An N tile count the ring cannot scatter evenly.
  if (clusterAxis.has_value()) {
    const ::ttnn::MeshShape &meshShape = context.getMeshDevice().shape();
    uint32_t ringSize = meshShape[static_cast<int32_t>(*clusterAxis)];
    LOG_ASSERT(ringSize > 2,
               "minimal_matmul_strided_reduce_scatter_async needs a "
               "reduce-scatter ring of more than 2 devices (a 2-device ring "
               "deadlocks in tt-metal's kernel); got ringSize=",
               ringSize);
    LOG_ASSERT(nTiles % ringSize == 0,
               "minimal_matmul_strided_reduce_scatter_async needs the N tile "
               "count (",
               nTiles, ") divisible by the reduce-scatter ring size (",
               ringSize, ") for an even scatter");
  }

  // N is split across the matmul grid's X axis, so grid.x must divide N tiles.
  uint32_t mmCoresX = largestDivisorAtMost(nTiles, gridX);

  // Reserve the bottom half of the compute grid for the reduce-scatter cores;
  // tt-metal sizes the RS worker count itself, so half the rows leaves room.
  LOG_ASSERT(gridY >= 2,
             "minimal_matmul_strided_reduce_scatter_async needs a compute grid "
             "with >= 2 rows; got gridY=",
             gridY);
  uint32_t availMMRows = std::max<uint32_t>(1, gridY / 2);

  // M is split across the matmul grid's Y axis, so grid.y must divide M tiles.
  uint32_t mmCoresY = largestDivisorAtMost(mTiles, availMMRows);

  // N tiles handled by each matmul core (N parallelized across grid.x). In
  // fused mode tt-metal splits this per-core width into blocks of a few tiles
  // and streams one block at a time to the RS; a single wide N block breaks the
  // RS chunking and the RS waits forever (deadlock). Match the reference
  // configs: cap the N block at 4 tiles and tell the RS how many blocks make up
  // a core's width via chunk_width_in_mm_blocks.
  uint32_t mmNFullBlockWt = nTiles / mmCoresX;
  uint32_t nBlock = largestDivisorAtMost(mmNFullBlockWt, 4);
  uint32_t chunkWidthInMmBlocks = mmNFullBlockWt / nBlock;

  ::ttnn::experimental::prim::MinimalMatmulConfig matmulConfig;
  // M/K blocks are likewise capped at 4 tiles and must divide their per-core
  // extents exactly.
  matmulConfig.M_block_size = largestDivisorAtMost(mTiles / mmCoresY, 4);
  matmulConfig.N_block_size = nBlock;
  matmulConfig.K_block_size = largestDivisorAtMost(kTiles, 4);
  matmulConfig.subblock_h = 1; // 1x1 subblock has no dst-register constraint
  matmulConfig.subblock_w = 1;
  matmulConfig.compute_with_storage_grid_size =
      ::tt::tt_metal::CoreCoord{mmCoresX, mmCoresY};

  // Reduce-scatter cores sit on the rows directly below the matmul grid
  // (offset convention from tt-metal's test: CoreCoord(0, mm_core_grid.y)).
  ::ttnn::CoreCoord reduceScatterCoreGridOffset{0, mmCoresY};

  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::experimental::minimal_matmul_strided_reduce_scatter_async(
          input, weight, static_cast<uint32_t>(op->dim()), multiDeviceSemaphore,
          reduceScatterCoreGridOffset, computeKernelConfig, numLinks,
          /*memory_config_mm=*/std::nullopt,
          /*rs_output_mem_config=*/memoryConfig,
          /*rs_intermediate_mem_config=*/std::nullopt, topology, clusterAxis,
          bias, /*fused_activation=*/std::nullopt, matmulConfig,
          barrierSemaphore, /*using_persistent_buffers=*/false,
          /*sub_device_id=*/std::nullopt,
          // Pass nullopt so tt-metal computes its own RS worker/buffer counts,
          // matching how its reference test invokes the kernel (forcing these
          // to 1 under-provisions the RS workers and deadlocks).
          /*num_workers_per_link=*/std::nullopt,
          /*num_buffers_per_channel=*/std::nullopt,
          /*chunk_width_in_mm_blocks=*/std::make_optional(chunkWidthInMmBlocks),
          /*optional_rs_output_tensor=*/std::nullopt,
          /*fused_ternary_scalar=*/scalar, addcmulInput1, addcmulInput2, dtype);

  // tt-metal returns two tensors: {matmul_intermediate [.., N], reduce_scatter
  // output [.., N/devices]} (see its reference test, which unpacks
  // `(tt_mm_out, tt_rs_out)`). The compiler only models the reduce-scatter
  // result, so bind that (the last output) to the single output ref; the matmul
  // intermediate is unused by the graph and its buffer is reclaimed here.
  const auto *outputRefs = op->outputs();
  LOG_ASSERT(outputs.size() == 2,
             "minimal_matmul_strided_reduce_scatter_async expected 2 outputs "
             "{matmul_intermediate, reduce_scatter}, got ",
             outputs.size());
  LOG_ASSERT(outputRefs->size() == 1,
             "minimal_matmul_strided_reduce_scatter_async flatbuffer expects 1 "
             "output, got ",
             outputRefs->size());
  tensorPool.insertTTNNTensorAndValidate(outputRefs->Get(0), outputs.back());
}
} // namespace tt::runtime::ttnn::operations::ccl
