// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormWidthShardInputRewritePattern.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// DistributedRMSNormOp op have the following limitations:
//
//  1. Input and residual tensors to be width-sharded in L1 with a ROW_MAJOR
//     shard spec (so the program factory can read the shard grid/shape).
//  2. The weight tensor in ROW_MAJOR layout with width equal to tile_width
//     (32). We reshape from 1D (N,) to 2D (N/32, 32) and convert layout.
//  3. A width-sharded output memory config matching the input shard spec
//     (skip_write_back = true). The fused kernel only all-gathers stats,
//     not data, so output shape == input shape.
//
// If the original output layout differs from the sharded config (e.g.
// interleaved DRAM), a to_memory_config op is inserted after the norm.
//
LogicalResult
DistributedRMSNormWidthShardInputRewritePattern::matchAndRewrite(
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

  // Create an affine map that translates the virtual grid layout to the
  // physical grid layout.
  auto affineMap = mlir::tt::ttnn::optimizer_utils::
      createSingleDeviceVirtualToPhysicalAffineMap(
          rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded,
          physicalGrid);
  auto grid = mlir::tt::ttcore::GridAttr::get(rewriter.getContext(),
                                              virtualGridSize, affineMap);
  auto memLayoutAttr = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded);

  // Create layout attribute for the input tensor with width-sharded L1 config.
  ttnn::TTNNLayoutAttr desiredInputLayout =
      ttnn::TTNNLayoutAttr::get(rewriter.getContext(), inputType.getShape(),
                                ttcore::TileType::get(inputElementType),
                                ttnn::BufferType::L1, grid, memLayoutAttr);

  if (currentInputLayout == desiredInputLayout) {
    return failure();
  }

  // Apply ToLayoutOp to convert the input tensor to width-sharded L1.
  ttnn::MemoryConfigAttr inputMemoryConfig =
      ttnn::MemoryConfigAttr::get(desiredInputLayout, grid);
  RankedTensorType memoryConfigedInputType =
      inputType.cloneWithEncoding(desiredInputLayout);
  auto inputToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
      op.getLoc(), memoryConfigedInputType, op.getInput(),
      tt::ttnn::Layout::Tile,
      ttcore::DataTypeAttr::get(
          rewriter.getContext(),
          ttcore::elementTypeToDataType(inputElementType)),
      inputMemoryConfig);

  // The runtime requires the weight tensor in ROW_MAJOR layout with
  // width = tile_width (32). Reshape from 1D (N,) to 2D (N/32, 32) first.
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

    // Reshape weight to 2D: (N,) -> (N/32, 32) so width matches tile width.
    int64_t totalElements = 1;
    for (int64_t dim : weightType.getShape()) {
      totalElements *= dim;
    }
    SmallVector<int64_t> reshapedShape = {totalElements / tileWidth, tileWidth};
    SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                          reshapedShape.end());
    RankedTensorType reshapedWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType,
                                                     reshapedShape);
    auto reshapeOp = rewriter.create<ttnn::ReshapeOp>(
        op.getLoc(), reshapedWeightType, weight,
        rewriter.getI32ArrayAttr(reshapedShapeI32), ttnn::MemoryConfigAttr());
    weight = reshapeOp.getResult();

    // Convert to ROW_MAJOR layout.
    weightType = mlir::cast<RankedTensorType>(weight.getType());
    weightLayout =
        mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(weightType.getEncoding());
    if (weightLayout && weightLayout.isTiled()) {
      ttnn::TTNNLayoutAttr rowMajorLayout =
          weightLayout.withLayout(ttnn::Layout::RowMajor,
                                  weightType.getShape());
      RankedTensorType rowMajorWeightType =
          weightType.cloneWithEncoding(rowMajorLayout);
      auto weightMemConfig = ttnn::MemoryConfigAttr::get(
          rewriter.getContext(),
          ttnn::TensorMemoryLayoutAttr::get(
              rewriter.getContext(),
              ttnn::TensorMemoryLayout::Interleaved),
          ttnn::BufferTypeAttr::get(rewriter.getContext(),
                                    weightLayout.getBufferType()),
          /*shardSpec=*/std::nullopt);
      auto weightToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), rowMajorWeightType, weight,
          tt::ttnn::Layout::RowMajor,
          ttcore::DataTypeAttr::get(
              rewriter.getContext(),
              ttcore::elementTypeToDataType(weightElementType)),
          weightMemConfig);
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
          op.getLoc(), shardedResidualType, residual,
          tt::ttnn::Layout::Tile,
          ttcore::DataTypeAttr::get(
              rewriter.getContext(),
              ttcore::elementTypeToDataType(inputElementType)),
          inputMemoryConfig);
      residual = residualToLayoutOp.getResult();
    }
  }

  // The fused kernel output shape == input shape (only stats are all-gathered,
  // not data). Use the input's width-sharded memory config for the output too
  // (skip_write_back = true in the program factory).
  RankedTensorType shardedOutputType =
      mlir::cast<RankedTensorType>(op.getResult().getType())
          .cloneWithEncoding(desiredInputLayout);

  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      op.getLoc(), shardedOutputType, inputToLayoutOp.getResult(), weight,
      residual, static_cast<uint32_t>(op.getClusterAxis()), op.getEpsilon(),
      op.getSubDeviceIdAttr(), inputMemoryConfig, op.getNumLinksAttr(),
      op.getTopologyAttr(), op.getComputeConfigAttr());

  // If the original output had a different layout (e.g. interleaved DRAM),
  // insert a to_memory_config to convert back.
  RankedTensorType originalOutputType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  auto originalOutputLayout = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
      originalOutputType.getEncoding());
  if (originalOutputLayout && originalOutputLayout != desiredInputLayout) {
    auto originalMemConfig = ttnn::MemoryConfigAttr::get(
        originalOutputLayout, originalOutputLayout.getGrid());
    auto toMemConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
        op.getLoc(), originalOutputType, newOp.getResult(), originalMemConfig);
    rewriter.replaceOp(op, toMemConfigOp.getResult());
  } else {
    rewriter.replaceOp(op, newOp.getResult());
  }
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
