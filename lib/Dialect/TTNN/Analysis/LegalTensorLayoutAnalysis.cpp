// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"

#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinOps.h"

#include <unordered_set>
#include <vector>

namespace mlir::tt::ttnn {

// Checks if we can create a TTNN tensor spec from the layout. If not, the
// layout is not compatible with the tensor shape and we discard it.
// On top of, check if we have enough tiles to shard the tensor to the
// desired grid. This is not necessarily a requirement, but it is a good
// heuristic to reduce the search space.
static bool tensorShapeCompatibleWithShard(RankedTensorType tensorType,
                                           TTNNLayoutAttr layout,
                                           GridAttr maxGrid) {

  if (!layout.hasShardedTensorMemoryLayout()) {
    return true;
  }

  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  if (!op_model::ttnn::isLayoutLegalForTensorShape(tensorShape, layout,
                                                   maxGrid)) {
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

// Function generates all possible TTNNLayout attributes for the given tensor
// type. maxNumGeneratedLayouts limits the number of generated layouts. If
// maxNumGeneratedLayouts is -1, all possible layouts are returned.
static std::vector<TTNNLayoutAttr> generateAllPossibleLayouts(
    mlir::MLIRContext *ctx, RankedTensorType tensorType, GridAttr maxGrid,
    Type scalarElementType, bool onlyShardedLayouts = false,
    int64_t maxNumGeneratedLayouts = -1, bool rowMajorAllowed = true) {

  std::vector<TTNNLayoutAttr> allPossibleLayouts;

  assert(tensorType.getEncoding() &&
         "Tensor type must have an encoding attribute.");
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  auto tensorShape = tensorType.getShape();
  Type tileElementType = TileType::get(scalarElementType);

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
        layoutAttr.withBufferType(BufferType::L1)
            .withMemoryLayout(TensorMemoryLayout::BlockSharded)
            .withElementType(elementType, tensorShape);

    // We can cache shard shape and then discard larger grids with same shard
    std::unordered_set<std::pair<int64_t, int64_t>,
                       llvm::pair_hash<int64_t, int64_t>>
        shardShapeSet;
    auto checkIfShardShapeExists = [&](llvm::ArrayRef<int64_t> shape) {
      assert(shape.size() == 2 && "Shard shape is expected to be 2D.");
      auto it = shardShapeSet.find({shape[0], shape[1]});
      if (it != shardShapeSet.end()) {
        return true;
      }
      shardShapeSet.insert({shape[0], shape[1]});
      return false;
    };

    assert(maxGrid.getShape().size() == 2 &&
           "Max device grid is expected to be 2D.");
    // Block Sharded
    auto affineMapBs =
        optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMap(
            ctx, TensorMemoryLayout::BlockSharded, maxGrid.getShape());
    for (int height = 1; height <= maxGrid.getShape()[0]; ++height) {
      for (int width = 1; width <= maxGrid.getShape()[1]; ++width) {
        shardedResults.push_back(
            shardedBase
                .withGrid(tensorType,
                          GridAttr::get(ctx, {height, width}, affineMapBs))
                .withMemoryLayout(TensorMemoryLayout::BlockSharded));
        if (checkIfShardShapeExists(
                shardedResults.back().getMemref().getShape())) {
          shardedResults.pop_back();
        }
      }
    }

    shardShapeSet.clear();

    int64_t numCores = maxGrid.getGridVolume();
    // Height Sharded
    auto affineMapHs =
        optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMap(
            ctx, TensorMemoryLayout::HeightSharded, maxGrid.getShape());

    for (int height = 1; height <= numCores; ++height) {
      shardedResults.push_back(
          shardedBase
              .withGrid(tensorType,
                        GridAttr::get(ctx, {height, 1}, affineMapHs))
              .withMemoryLayout(TensorMemoryLayout::HeightSharded));

      if (checkIfShardShapeExists(
              shardedResults.back().getMemref().getShape())) {
        shardedResults.pop_back();
      }
    }

    shardShapeSet.clear();

    // Width Sharded
    auto affineMapWs =
        optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMap(
            ctx, TensorMemoryLayout::WidthSharded, maxGrid.getShape());
    for (int width = 1; width <= numCores; ++width) {
      shardedResults.push_back(
          shardedBase
              .withGrid(tensorType, GridAttr::get(ctx, {1, width}, affineMapWs))
              .withMemoryLayout(TensorMemoryLayout::WidthSharded));
      if (checkIfShardShapeExists(
              shardedResults.back().getMemref().getShape())) {
        shardedResults.pop_back();
      }
    }
  }

  // Filter layouts based on output tensor legality for current op.
  shardedResults.erase(
      std::remove_if(shardedResults.begin(), shardedResults.end(),
                     [tensorType, maxGrid](TTNNLayoutAttr layout) {
                       return !tensorShapeCompatibleWithShard(tensorType,
                                                              layout, maxGrid);
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

// ===----------------------------------------------------------------------===//
// LegalTensorLayoutAnalysis
// ===----------------------------------------------------------------------===//

void LegalTensorLayoutAnalysis::analysisImplementation() {
  mlir::ModuleOp moduleOp = mlir::cast<mlir::ModuleOp>(op);
  llvm::DenseSet<RankedTensorType> processedTypes;

  // Walk through module and collect all tensor types
  moduleOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      auto tensorType = mlir::cast<RankedTensorType>(operand.getType());
      // Only process each unique tensor type once
      if (processedTypes.insert(tensorType).second) {
        processTensorType(tensorType);
      }
    }

    for (Value result : op->getResults()) {
      if (!mlir::isa<RankedTensorType>(result.getType())) {
        continue;
      }
      auto tensorType = mlir::cast<RankedTensorType>(result.getType());
      // Only process each unique tensor type once
      if (processedTypes.insert(tensorType).second) {
        processTensorType(tensorType);
      }
    }
  });
}

void LegalTensorLayoutAnalysis::processTensorType(RankedTensorType tensorType) {
  // Generate all possible layouts for this tensor type
  std::vector<TTNNLayoutAttr> layouts = generateLayouts(tensorType);

  // Categorize layouts by scalar type, memory layout, and data layout
  for (const TTNNLayoutAttr &layout : layouts) {
    Type scalarType = layout.getScalarElementType();

    TensorMemoryLayout memLayout = TensorMemoryLayout::Interleaved;
    if (TensorMemoryLayoutAttr memLayoutAttr = layout.getMemLayout()) {
      memLayout = memLayoutAttr.getValue();
    }

    std::size_t pageLayoutIndex = getPageLayoutIndex(layout.getLayout());
    std::size_t memLayoutIndex = getMemoryLayoutIndex(memLayout);

    // Store layout in the categorized map
    analysisResult[tensorType][scalarType][pageLayoutIndex][memLayoutIndex]
        .push_back(layout);
  }
}

std::vector<TTNNLayoutAttr>
LegalTensorLayoutAnalysis::generateLayouts(RankedTensorType tensorType) {

  std::vector<TTNNLayoutAttr> allLayouts;

  // Assert that we have allowedScalarTypes and it's not empty
  assert(analysisInput.allowedScalarTypes &&
         "LegalTensorAnalysis requires allowedScalarTypes to be non-null");
  assert(!analysisInput.allowedScalarTypes->empty() &&
         "LegalTensorAnalysis requires at least one scalar type");

  // Generate layouts for each allowed scalar type
  for (Type scalarType : *analysisInput.allowedScalarTypes) {
    auto layoutsForType = generateAllPossibleLayouts(
        tensorType.getContext(), tensorType, analysisInput.maxGrid, scalarType,
        /*onlyShardedLayouts*/ false, /*maxNumGeneratedLayouts*/ -1,
        analysisInput.rowMajorAllowed);

    allLayouts.insert(allLayouts.end(), layoutsForType.begin(),
                      layoutsForType.end());
  }

  return allLayouts;
}

} // namespace mlir::tt::ttnn
