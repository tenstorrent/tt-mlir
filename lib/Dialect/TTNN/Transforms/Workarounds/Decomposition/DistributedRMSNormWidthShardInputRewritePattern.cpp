// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormWidthShardInputRewritePattern.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::tt::ttnn::workarounds::decomposition {

// Prepares DistributedRMSNormOp for the tt-metal fused_rms_minimal kernel
// at the layout / sharding level. This pattern only handles operand layout
// changes; it does not allocate scratch buffers or global semaphores. Those
// are handled by the dedicated TTNNAllocateDistributedOpBuffers and
// TTNNAllocateDistributedOpSemaphores passes (see DistributedOpInterface).
// Weight 1D->2D ReshapeOps are emitted in
// DistributedRMSNormDecompositionRewritePattern alongside the input reshape.
//
// What this pattern enforces:
//
//  1. Input and residual tensors must be width-sharded in L1 with a ROW_MAJOR
//     shard spec (so the program factory can read the shard grid/shape).
//  2. The weight tensor (assumed already 2D after decomposition) must be in
//     ROW_MAJOR layout. We insert a ToLayoutOp if it is currently tiled.
//  3. The output memory config must match the input shard spec
//     (skip_write_back = true). The fused kernel only all-gathers stats,
//     not data, so output shape == input shape.
//  4. A compute_config must be set. If absent, a default HiFi4 config with
//     fp32_dest_acc_en=true is created. Downstream passes (in particular
//     TTNNAllocateDistributedOpBuffers) read fp32_dest_acc_en to derive
//     the stats scratch dtype.
//  5. A LayerNormShardedMultiCoreProgramConfig is computed from the input's
//     shard spec (grid size, block_h, block_w) and attached as an attribute.
//
// If the original output layout differs from the sharded config (e.g.
// interleaved DRAM), a to_memory_config op is inserted after the norm.
//
LogicalResult DistributedRMSNormWidthShardInputRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());

  // Check if the input is already width-sharded in L1.
  ttnn::TTNNLayoutAttr currentInputLayout =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  if (currentInputLayout) {
    auto memLayout = currentInputLayout.getMemLayout();
    if (memLayout &&
        memLayout.getValue() == ttnn::TensorMemoryLayout::WidthSharded) {
      return failure();
    }
  }

  auto inputElementType = inputType.getElementType();
  if (ttcore::TileType inputTileType =
          mlir::dyn_cast_or_null<ttcore::TileType>(inputElementType)) {
    inputElementType = inputTileType.getElementType();
  }

  // Compute the number of tiles along the width (last) dimension.
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t tileWidth = 32;
  int64_t numWidthTiles = (inputShape.back() + tileWidth - 1) / tileWidth;

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op.getOperation());

  // Retrieve the physical worker grid ([H, W]) for the device. We use the
  // worker grid (not the raw chip grid) because that is what canonical core
  // placement and the runtime use, so the core-count choice and the actual
  // placement stay consistent.
  llvm::ArrayRef<int64_t> workerGridShape =
      deviceAttr.getWorkerGrid().getShape();
  int64_t physGridH = workerGridShape[0];
  int64_t physGridW = workerGridShape[1];
  int64_t maxCores = physGridH * physGridW;

  // Choose the core grid for the width-sharded input.
  //
  // #5738 root cause: the fused rms_allgather kernel multicasts and reduces
  // stats across the *bounding-box rectangle* of the shard cores
  // (num_mcast_dests = grid_w * grid_h; see tt-metal
  // rms_allgather_program_factory.cpp). If the shard cores do not exactly fill
  // that rectangle, the extra "phantom" cores in the bounding box hold
  // uninitialized L1 yet still participate in the reduction -> the norm reads
  // uninitialized memory and the whole decode explodes into NaN/1e36 garbage.
  //
  // The previous logic picked the largest core count dividing numWidthTiles
  // and relied on canonical row-major placement, which wraps at the physical
  // grid *width*. The failure is the wrapping, not the core count: for a 70B
  // decode norm numWidthTiles = 4096/32 = 128, so the count is 64. On Wormhole
  // (8x8 worker grid) 64 shards wrap at width 8 into a perfect 8x8 rectangle,
  // so it happened to work. On Blackhole (11x10 worker grid) 64 shards wrap at
  // width 11 into 5 full rows (55 cores) + 9 cores, a non-rectangular set whose
  // bounding box is 11x6 = 66 cores with 2 phantom cores at (9,5) and (10,5)
  // -> uninitialized read. (64 itself fits fine as a rectangle in 11x10; only
  // the row-major-at-full-width placement makes it non-rectangular.)
  //
  // To be correct on every architecture we pick a *rectangular* core grid
  // (gridW x gridH) that (a) fits in the physical worker grid, (b) has a core
  // count dividing numWidthTiles (so the width shards evenly), and (c)
  // maximizes the core count for throughput, then place it explicitly. The
  // shard cores then exactly fill their bounding box, so there are no phantom
  // cores. On Blackhole this yields 8x8 = 64 cores (same count as before, just
  // rectangular -> no throughput loss); on Wormhole it is unchanged at 8x8.
  int64_t rectW = 1, rectH = 1, numCores = 1;
  for (int64_t h = 1; h <= physGridH; ++h) {
    for (int64_t w = 1; w <= physGridW; ++w) {
      int64_t cores = w * h;
      if (cores > maxCores || cores > numWidthTiles) {
        continue;
      }
      if (numWidthTiles % cores != 0) {
        continue;
      }
      if (cores > numCores) {
        numCores = cores;
        rectW = w;
        rectH = h;
      }
    }
  }

  // Width-sharded layouts use a [1, numCores] virtual grid; the physical
  // placement is an explicit gridW x gridH rectangle at the origin so the
  // shard cores fill their bounding box exactly (no phantom cores). We set the
  // CoreRangeSet explicitly instead of using canonical row-major placement,
  // which would wrap at the physical grid width and re-introduce a
  // non-rectangular layout for core counts that are not a multiple of the
  // grid width.
  SmallVector<int64_t> virtualGridSize = {1, numCores};
  ttnn::CoreRangeSetAttr shardCoreRangeSet = ttnn::CoreRangeSetAttr::get(
      rewriter.getContext(),
      ttnn::CoreRangeAttr::get(
          rewriter.getContext(),
          ttnn::CoreCoordAttr::get(rewriter.getContext(), 0, 0),
          ttnn::CoreCoordAttr::get(rewriter.getContext(), rectW - 1,
                                   rectH - 1)));

  // Create layout attribute for the input tensor with width-sharded L1 config.
  ttnn::TTNNLayoutAttr desiredInputLayout =
      ttnn::TTNNLayoutAttr::Builder(rewriter.getContext(), inputType.getShape(),
                                    ttcore::TileType::get(inputElementType))
          .setBufferType(ttnn::BufferType::L1)
          .setMemoryLayout(ttnn::TensorMemoryLayout::WidthSharded)
          .setGridShape(virtualGridSize)
          .setCoreRangeSet(shardCoreRangeSet)
          .build();

  if (currentInputLayout == desiredInputLayout) {
    return failure();
  }

  // Apply ToLayoutOp to convert the input tensor to width-sharded L1.
  RankedTensorType memoryConfigedInputType =
      inputType.cloneWithEncoding(desiredInputLayout);
  auto inputToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
      op.getLoc(), memoryConfigedInputType, op.getInput());

  // The fused kernel requires the weight in ROW_MAJOR layout. The 1D->2D
  // shape reshape is handled by the decomposition pattern; here we only
  // convert from Tile to RowMajor if needed.
  mlir::Value weight = op.getWeight();
  if (weight) {
    RankedTensorType weightType =
        mlir::cast<RankedTensorType>(weight.getType());
    ttnn::TTNNLayoutAttr weightLayout =
        mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(weightType.getEncoding());

    auto weightElementType = weightType.getElementType();
    if (ttcore::TileType weightTileType =
            mlir::dyn_cast_or_null<ttcore::TileType>(weightElementType)) {
      weightElementType = weightTileType.getElementType();
    }

    if (weightLayout && weightLayout.isTiled()) {
      ttnn::TTNNLayoutAttr rowMajorLayout =
          ttnn::TTNNLayoutAttr::Builder(weightLayout, weightType.getShape())
              .setLayout(ttnn::Layout::RowMajor);
      RankedTensorType rowMajorWeightType =
          weightType.cloneWithEncoding(rowMajorLayout);
      auto weightToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), rowMajorWeightType, weight);
      weight = weightToLayoutOp.getResult();
    }
  }

  // The runtime requires the residual to have the same shard spec as input.
  mlir::Value residual = op.getResidual();
  if (residual) {
    RankedTensorType residualType =
        mlir::cast<RankedTensorType>(residual.getType());
    ttnn::TTNNLayoutAttr residualLayout =
        mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
            residualType.getEncoding());
    if (residualLayout != desiredInputLayout) {
      RankedTensorType shardedResidualType =
          residualType.cloneWithEncoding(desiredInputLayout);
      auto residualToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), shardedResidualType, residual);
      residual = residualToLayoutOp.getResult();
    }
  }

  // Ensure compute_config is set. The buffer-allocation pass derives the
  // stats scratch dtype from compute_config.fp32_dest_acc_en, so the
  // attribute must be populated before that pass runs.
  auto computeConfigAttr = op.getComputeConfigAttr();
  if (!computeConfigAttr) {
    computeConfigAttr = DeviceComputeKernelConfigAttr::get(
        rewriter.getContext(),
        /*mathFidelity=*/MathFidelity::HiFi4,
        /*mathApproxMode=*/BoolAttr::get(rewriter.getContext(), false),
        /*fp32DestAccEn=*/BoolAttr::get(rewriter.getContext(), true),
        /*packerL1Acc=*/BoolAttr::get(rewriter.getContext(), true),
        /*dstFullSyncEn=*/nullptr);
  }

  // The fused kernel output shape == input shape (only stats are all-gathered,
  // not data). Use the input's width-sharded memory config for the output too
  // (skip_write_back = true in the program factory).
  RankedTensorType shardedOutputType =
      mlir::cast<RankedTensorType>(op.getResult().getType())
          .cloneWithEncoding(desiredInputLayout);

  // Compute LayerNormShardedMultiCoreProgramConfig from the shard spec.
  // This mirrors ttnn::prim::create_program_config(shardSpec).
  auto scalarShardShape = desiredInputLayout.getScalarShardShape();
  int64_t blockH = scalarShardShape[0] / tileWidth;
  int64_t blockW = scalarShardShape[1] / tileWidth;

  // Derive the kernel's compute grid from the bounding box of the input's
  // shard cores. The semaphore-allocation pass independently derives the
  // semaphore core range from the same shard spec at a later pipeline
  // stage (post-optimizer); see DistributedRMSNormOp::allocateSemaphores.
  auto inputMemoryConfig = ttnn::MemoryConfigAttr::get(desiredInputLayout);
  std::optional<ttnn::ShardSpecAttr> inputShardSpec =
      inputMemoryConfig.getShardSpec();
  assert(inputShardSpec.has_value() &&
         "width-sharded input must have a shard spec");
  std::optional<ttnn::CoreRangeAttr> inputBoundingBox =
      inputShardSpec->getCoreRangeSet().getBoundingBox();
  assert(inputBoundingBox.has_value() &&
         "width-sharded input shard spec must have at least one core range");
  ttnn::CoreCoordAttr boxStart = inputBoundingBox->getStartCoord();
  ttnn::CoreCoordAttr boxEnd = inputBoundingBox->getEndCoord();

  // Same as tt-metal GridParams: gs = bbox.end - bbox.start + 1 (inclusive
  // bounding box of the input shard cores).
  uint64_t gridW = boxEnd.getX() - boxStart.getX() + 1;
  uint64_t gridH = boxEnd.getY() - boxStart.getY() + 1;
  auto programConfigAttr =
      ttnn::LayerNormShardedMultiCoreProgramConfigAttr::get(
          rewriter.getContext(),
          ttnn::CoreCoordAttr::get(rewriter.getContext(), gridW, gridH),
          /*subblock_w=*/1, blockH, blockW, /*inplace=*/false);

  // Rebuild the op with the converted operands and updated attributes.
  // The stats and semaphore operands stay null here; they are populated by
  // the TTNNAllocateDistributedOpBuffers / TTNNAllocateDistributedOpSemaphores
  // passes (see DistributedOpInterface implementation in TTNNOps.cpp).
  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      op.getLoc(), shardedOutputType, inputToLayoutOp.getResult(), weight,
      residual, /*stats=*/nullptr, /*semaphore=*/nullptr, op.getDevice(),
      static_cast<uint32_t>(op.getClusterAxis()), op.getEpsilon(),
      op.getSubDeviceIdAttr(), op.getNumLinksAttr(), op.getTopologyAttr(),
      computeConfigAttr, programConfigAttr);

  // If the original output had a different layout (e.g. interleaved DRAM),
  // insert a to_memory_config to convert back.
  RankedTensorType originalOutputType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  auto originalOutputLayout = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
      originalOutputType.getEncoding());
  if (originalOutputLayout && originalOutputLayout != desiredInputLayout) {
    auto toMemConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
        op.getLoc(), originalOutputType, newOp.getResult());
    rewriter.replaceOp(op, toMemConfigOp.getResult());
  } else {
    rewriter.replaceOp(op, newOp.getResult());
  }
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
