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

  // Retrieve the physical grid shape for the device.
  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0].getGrid();
  int64_t maxCores =
      physicalGrid[0] * physicalGrid[1]; // total cores in physical grid

  // Pick the largest core count that evenly divides numWidthTiles,
  // capped by the physical grid size.
  int64_t numCores = 1;
  for (int64_t c = std::min(maxCores, numWidthTiles); c >= 1; --c) {
    if (numWidthTiles % c == 0) {
      numCores = c;
      break;
    }
  }
  SmallVector<int64_t> virtualGridSize = {1, numCores};

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op.getOperation());

  // Create layout attribute for the input tensor with width-sharded L1 config.
  ttnn::TTNNLayoutAttr desiredInputLayout =
      ttnn::TTNNLayoutAttr::Builder(rewriter.getContext(), inputType.getShape(),
                                    ttcore::TileType::get(inputElementType))
          .setBufferType(ttnn::BufferType::L1)
          .setMemoryLayout(ttnn::TensorMemoryLayout::WidthSharded)
          .setGridShape(virtualGridSize)
          .buildWithCanonicalCorePlacement(deviceAttr);

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
