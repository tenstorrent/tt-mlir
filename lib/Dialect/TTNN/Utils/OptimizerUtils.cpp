// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttnn::optimizer_utils {

AffineMap createSingleDeviceVirtualToPhysicalAffineMap(
    MLIRContext *context,
    const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
    const llvm::ArrayRef<int64_t> physicalGridShape) {

  AffineExpr workerDeviceIdx = mlir::getAffineConstantExpr(0, context);

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded: {
    // Create affine map that maps width sharded virtual grid 1xN to the
    // physical grid gridShape[0] x gridShape[1]
    AffineExpr virtualWidth = mlir::getAffineDimExpr(1, context); // d1
    AffineExpr workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    AffineMap widthMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualWidth.floorDiv(workerCoreW),
         virtualWidth % workerCoreW},
        context);
    return widthMap;
  }
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded: {
    // Create affine map that maps height sharded virtual grid Mx1 to the
    // physical grid gridShape[0] x gridShape[1]
    AffineExpr virtualHeight = mlir::getAffineDimExpr(0, context); // d0
    AffineExpr workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    AffineMap heightMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualHeight.floorDiv(workerCoreW),
         virtualHeight % workerCoreW},
        context);
    return heightMap;
  }
  default:
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded: {
    AffineExpr d0 = mlir::getAffineDimExpr(0, context); // d0
    AffineExpr d1 = mlir::getAffineDimExpr(1, context); // d1
    AffineMap blockMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {workerDeviceIdx, d0, d1}, context);
    return blockMap;
  }
  }
}

static bool tensorShapeCompatibleWithShard(RankedTensorType tensorType,
                                           TTNNLayoutAttr layout) {

  // Check if we can create a TTNN tensor spec from the layout. If not, the
  // layout is not compatible with the tensor shape and we discard it.
  // On top of, check if we have enough tiles to shard the tensor to the
  // desired grid. This is not necessarily a requirement, but it is a good
  // heuristic to reduce the search space.
  if (!layout.hasShardedTensorMemoryLayout()) {
    return true;
  }

  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  if (!op_model::ttnn::isLayoutLegalForTensorShape(tensorShape, layout)) {
    return false;
  }

  // TODO(rpavlovicTT): Revisit this logic now that we are able to check
  // validity through op model interface. If the check is removed, we create
  // more layouts, but for some reason this introduces test failures. Need to
  // investigate if it may be safely removed.
  if (layout.isTiled()) {
    llvm::SmallVector<int64_t, 2> tiledShape =
        layout.getTiledShape(tensorShape);
    llvm::ArrayRef<int64_t> gridShape = layout.getGrid().getShape();

    assert(tiledShape.size() == gridShape.size() &&
           "Tiled tensor shape and grid shape must have the same rank");

    for (size_t i = 0; i < tiledShape.size(); i++) {
      // We need to have at least as many tiles as the grid size.
      // Could also experiment with tiledShape[i] % gridShape[i] == 0, but need
      // more context.
      if (tiledShape[i] < gridShape[i]) {
        return false;
      }
    }
    return true;
  }

  // TODO(odjuricic): For row major there are no constraints on how the tensor
  // can be sharded. We need some kind of a heuristic to reduce the search
  // space.
  return true;
}

std::vector<TTNNLayoutAttr> generateAllPossibleLayouts(
    mlir::MLIRContext *ctx, RankedTensorType tensorType, GridAttr maxGrid,
    Type scalarElementType, bool onlyShardedLayouts,
    int64_t maxNumGeneratedLayouts, bool rowMajorAllowed) {

  std::vector<TTNNLayoutAttr> allPossibleLayouts;

  assert(tensorType.getEncoding() &&
         "Tensor type must have an encoding attribute.");
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  auto tensorShape = tensorType.getShape();
  Type tileElementType = TileType::get(ctx, scalarElementType);

  std::vector<TTNNLayoutAttr> shardedResults;

  for (auto elementType : {scalarElementType, tileElementType}) {
    if (!rowMajorAllowed && elementType == scalarElementType) {
      continue;
    }

    if (!onlyShardedLayouts) {
      // DRAM
      allPossibleLayouts.push_back(TTNNLayoutAttr::get(
          ctx, tensorShape, elementType, BufferType::DRAM, maxGrid,
          TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::Interleaved)));

      // L1 Interleaved - It must be tiled.
      // TODO(odjuricic): Check that this is always the case.
      if (mlir::isa<TileType>(elementType)) {
        allPossibleLayouts.push_back(TTNNLayoutAttr::get(
            ctx, tensorShape, elementType, BufferType::L1, maxGrid,
            TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::Interleaved)));
      }
    }

    // L1 Sharded
    TTNNLayoutAttr shardedBase =
        layoutAttr.withBufferType(ctx, BufferType::L1)
            .withMemoryLayout(ctx, TensorMemoryLayout::BlockSharded)
            .withElementType(ctx, elementType, tensorShape);

    assert(maxGrid.getShape().size() == 2 &&
           "Max device grid is expected to be 2D.");
    // Block Sharded
    auto affineMapBs = createSingleDeviceVirtualToPhysicalAffineMap(
        ctx, TensorMemoryLayout::BlockSharded, maxGrid.getShape());
    for (int height = 1; height <= maxGrid.getShape()[0]; ++height) {
      for (int width = 1; width <= maxGrid.getShape()[1]; ++width) {
        shardedResults.push_back(
            shardedBase
                .withGrid(ctx, tensorType,
                          GridAttr::get(ctx, {height, width}, affineMapBs))
                .withMemoryLayout(ctx, TensorMemoryLayout::BlockSharded));
      }
    }

    int64_t numCores = maxGrid.getGridVolume();
    // Height Sharded
    auto affineMapHs = createSingleDeviceVirtualToPhysicalAffineMap(
        ctx, TensorMemoryLayout::HeightSharded, maxGrid.getShape());

    for (int height = 1; height <= numCores; ++height) {
      shardedResults.push_back(
          shardedBase
              .withGrid(ctx, tensorType,
                        GridAttr::get(ctx, {height, 1}, affineMapHs))
              .withMemoryLayout(ctx, TensorMemoryLayout::HeightSharded));
    }

    // Width Sharded
    auto affineMapWs = createSingleDeviceVirtualToPhysicalAffineMap(
        ctx, TensorMemoryLayout::WidthSharded, maxGrid.getShape());
    for (int width = 1; width <= numCores; ++width) {
      shardedResults.push_back(
          shardedBase
              .withGrid(ctx, tensorType,
                        GridAttr::get(ctx, {1, width}, affineMapWs))
              .withMemoryLayout(ctx, TensorMemoryLayout::WidthSharded));
    }
  }

  // Filter layouts based on output tensor legality for current op.
  shardedResults.erase(
      std::remove_if(shardedResults.begin(), shardedResults.end(),
                     [tensorType](TTNNLayoutAttr layout) {
                       return !tensorShapeCompatibleWithShard(tensorType,
                                                              layout);
                     }),
      shardedResults.end());

  // Pick top largest sharded grids.
  // This becomes a problem when we introduce row_major since an 8x8 tensor can
  // be sharded onto a 8x8 grid.
  std::sort(shardedResults.begin(), shardedResults.end(),
            [](TTNNLayoutAttr a, TTNNLayoutAttr b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });

  const int64_t numLayoutsToGenerate =
      maxNumGeneratedLayouts >= 0
          ? std::min(maxNumGeneratedLayouts,
                     static_cast<int64_t>(shardedResults.size()))
          : shardedResults.size();

  allPossibleLayouts.insert(allPossibleLayouts.end(), shardedResults.begin(),
                            shardedResults.begin() + numLayoutsToGenerate);

  return allPossibleLayouts;
}

} // namespace mlir::tt::ttnn::optimizer_utils
